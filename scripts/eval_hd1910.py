"""Evaluate or view exported HD1910 policies in the matching training physics.

This is a flat-ground test: pushes, mass/CoM/friction DR and observation noise
are disabled; the HD1910 BAM voltage distribution and command delays remain.
"""

import argparse
import json
from pathlib import Path
import queue

import numpy as np
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


CASES = [
    ("stand", (0.0, 0.0, 0.0)),
    ("forward", (0.15, 0.0, 0.0)),
    ("backward", (-0.15, 0.0, 0.0)),
    ("left", (0.0, 0.10, 0.0)),
    ("right", (0.0, -0.10, 0.0)),
    ("turn_left", (0.0, 0.0, 0.6)),
    ("turn_right", (0.0, 0.0, -0.6)),
]


def make_env(num_envs):
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-HD1910", play=True)
    cfg.scene.num_envs = num_envs
    keep = {"reset_base", "reset_robot_joints", "reset_action_history", "expand_bam_friction_fields"}
    cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
    cfg.curriculum = {}
    cfg.observations["actor"].enable_corruption = False
    for term in cfg.observations["actor"].terms.values():
        if "max_angle_deg" in term.params:
            term.params["max_angle_deg"] = 0.0
    for term in cfg.commands.values():
        term.resampling_time_range = (1000.0, 1000.0)
    cfg.viewer.distance = 0.7
    return ManagerBasedRlEnv(cfg, device="cuda:0")


class OnnxPolicy:
    def __init__(self, env, path):
        self.env = env
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, obs, commands):
        for name, value in (("twist", commands), ("head_pose", 0.0), ("body_pose", 0.0)):
            self.env.command_manager.get_command(name)[:] = value
        actor = obs["actor"].detach().cpu().numpy().copy()
        # Replace cached command observations without advancing sensor histories.
        actor[:, 48:51] = commands.cpu().numpy()
        actor[:, 51:61] = 0.0
        # Deployment exports intentionally have a fixed batch dimension of one.
        actor = actor.astype(np.float32)
        actions = np.concatenate([
            self.session.run(None, {self.input: row[None, :]})[0] for row in actor
        ], axis=0)
        if not np.isfinite(actions).all():
            raise RuntimeError("ONNX returned a non-finite action")
        return torch.as_tensor(actions, device=self.env.device)


def evaluate(args):
    n = len(CASES) * args.repeats
    env = make_env(n)
    try:
        policy = OnnxPolicy(env, args.onnx)
        commands = torch.tensor([v for _, v in CASES], device=env.device).repeat_interleave(args.repeats, 0)
        obs, _ = env.reset(seed=42)
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        sums = torch.zeros((n, 3), device=env.device)
        counts = torch.zeros(n, device=env.device)
        max_tilt = torch.zeros(n, device=env.device)
        for step in range(round(args.seconds / env.step_dt)):
            obs, _, terminated, truncated, _ = env.step(policy(obs, commands))
            alive &= ~(terminated | truncated)
            robot = env.scene["robot"]
            velocity = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), 1)
            gravity = obs["critic"][:, 6:9]
            tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1)))
            max_tilt = torch.maximum(max_tilt, torch.where(alive, tilt, 0))
            if step >= 100:  # Exclude the first two seconds of settling.
                sums += torch.where(alive[:, None], velocity, 0)
                counts += alive
        rows = []
        for i, (name, command) in enumerate(CASES):
            sl = slice(i * args.repeats, (i + 1) * args.repeats)
            count = counts[sl].sum()
            mean = (sums[sl].sum(0) / count.clamp_min(1)).cpu().tolist()
            row = dict(case=name, command=list(command), mean_actual_vx_vy_wz=mean,
                       survived=int(alive[sl].sum()), trials=args.repeats,
                       max_tilt_deg=float(max_tilt[sl].max()))
            rows.append(row)
            print(json.dumps(row), flush=True)
        output = dict(policy=str(args.onnx), seconds=args.seconds, settling_seconds=2,
                      note="Nominal simulation only. Mean speeds exclude post-fall samples and initial 2s.", cases=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    finally:
        env.close()


def view(args):
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import NativeMujocoViewer

    env = make_env(1)
    wrapped = RslRlVecEnvWrapper(env)
    policy = OnnxPolicy(env, args.onnx)
    requests = queue.Queue()
    selected = 0
    commands = torch.zeros((1, 3), device=env.device)
    auto_demo = args.demo
    demo_segments = [(0, 2), (3, 6), (0, 2), (4, 6), (0, 2), (1, 6), (2, 6), (5, 6), (6, 6)]
    segment = 0
    segment_steps = 0

    def key_callback(key):
        if ord("1") <= key <= ord("7"):
            requests.put(key - ord("1"))
        elif key == ord("0"):
            requests.put(-1)

    def infer(obs):
        nonlocal selected, auto_demo, segment, segment_steps
        changed = False
        while not requests.empty():
            request = requests.get_nowait()
            if request == -1:
                auto_demo = not auto_demo
                segment, segment_steps = 0, 0
                selected = 0
            else:
                selected = request
                auto_demo = False
            changed = True
        if auto_demo:
            segment_steps += 1
            if segment_steps >= round(demo_segments[segment][1] / env.step_dt):
                segment = (segment + 1) % len(demo_segments)
                segment_steps = 0
                selected = demo_segments[segment][0]
                changed = True
        if changed:
            obs, _ = wrapped.reset()
            print("Selected:", selected + 1, CASES[selected], flush=True)
        commands[:] = torch.tensor(CASES[selected][1], device=env.device)
        return policy(obs, commands)

    try:
        wrapped.reset()
        print("Click the VIEWER; keys 1..7: stand, forward, backward, left, right, turn left, turn right.", flush=True)
        print("0 toggles the automatic demo. Each selected segment starts from a fresh reset.", flush=True)
        class ReportedViewer(NativeMujocoViewer):
            def _set_status_overlay(self, viewer):
                import mujoco
                velocity = env.scene["robot"].data.root_link_lin_vel_b[0].cpu().tolist()
                yaw_rate = float(env.scene["robot"].data.root_link_ang_vel_b[0, 2])
                viewer.set_texts((
                    mujoco.mjtFontScale.mjFONTSCALE_150.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                    "HD1910 walking demo\nAction\nTarget vx / vy / wz\nActual vx / vy / wz\nControls\nMode",
                    f"ONNX policy\n{CASES[selected][0]}\n"
                    f"{CASES[selected][1][0]:+.2f} / {CASES[selected][1][1]:+.2f} / {CASES[selected][1][2]:+.2f}\n"
                    f"{velocity[0]:+.2f} / {velocity[1]:+.2f} / {yaw_rate:+.2f}\n"
                    f"1..7 select | 0 auto | Space pause\n{'AUTO (reset per segment)' if auto_demo else 'MANUAL'}",
                ))

            def setup(self):
                super().setup()
                self._visible_geom_groups = self.viewer.opt.geomgroup.copy()
                Path("logs/hd1910_viewer_status.json").write_text(
                    json.dumps(dict(status="open", policy=str(args.onnx))), encoding="utf-8"
                )

            def sync_env_to_viewer(self):
                # MuJoCo also binds number keys to geometry visibility. Restore
                # the initial groups so selecting an action cannot hide the robot.
                groups = getattr(self, "_visible_geom_groups", None)
                if groups is not None and self.viewer is not None:
                    self.viewer.opt.geomgroup[:] = groups
                super().sync_env_to_viewer()

            def close(self):
                super().close()
                Path("logs/hd1910_viewer_status.json").write_text(
                    json.dumps(dict(status="closed", policy=str(args.onnx))), encoding="utf-8"
                )

        ReportedViewer(wrapped, infer, key_callback=key_callback).run()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("models/hd1910_velocity_final.onnx"))
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Open MuJoCo with an automatic walking/sideways demo; keys 1..7 take manual control")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("logs/hd1910_final_eval.json"))
    args = parser.parse_args()
    if args.seconds <= 2 or args.repeats < 1:
        parser.error("seconds must exceed 2 and repeats must be positive")
    (view if args.view or args.demo else evaluate)(args)


if __name__ == "__main__":
    main()
