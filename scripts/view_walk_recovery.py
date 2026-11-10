"""Interactive HD1910 walking/recovery supervisor using two exported policies."""

import argparse
import json
from pathlib import Path
import queue

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer


class Supervisor:
    def __init__(self, env, walk, recovery):
        self.env = env
        self.sessions = {}
        for name, path in [('WALK', walk), ('RECOVERY', recovery)]:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
            if session.get_inputs()[0].shape[-1] != 61 or session.get_outputs()[0].shape[-1] != 14:
                raise ValueError(f'Incompatible policy: {path}')
            self.sessions[name] = session
        self.requests = queue.Queue()
        self.rng = np.random.default_rng(42)
        self.maximum_push = 0.7
        self.pushes = 0
        self.transitions = []
        self.reset()

    def reset(self):
        self.command = np.zeros(3, dtype=np.float32)
        self.mode = 'WALK'
        self.bad_time = self.stable_time = self.recovery_time = 0.0
        self.tilt = self.height = 0.0
        self.steps = 0

    def set_mode(self, mode):
        if self.mode != mode:
            self.transitions.append(dict(step=self.steps, source=self.mode, target=mode))
            self.mode = mode
            self.stable_time = self.recovery_time = self.bad_time = 0.0
            print(f'Controller: {mode}', flush=True)

    def status(self):
        return dict(mode=self.mode, command=self.command.tolist(), tilt=self.tilt,
                    height=self.height, pushes=self.pushes, push_speed=self.maximum_push,
                    steps=self.steps, transitions=self.transitions)

    def __call__(self, obs):
        commands = {
            ord('W'): (0.15, 0, 0), ord('S'): (-0.15, 0, 0),
            ord('A'): (0, 0.10, 0), ord('D'): (0, -0.10, 0),
            ord('Q'): (0, 0, 0.6), ord('E'): (0, 0, -0.6),
            ord('X'): (0, 0, 0),
        }
        while not self.requests.empty():
            key = self.requests.get_nowait()
            if key in commands:
                self.command[:] = commands[key]
                print(f'Requested vx/vy/yaw: {self.command.tolist()}', flush=True)
            elif key == ord('P'):
                angle = self.rng.uniform(0, 2 * np.pi)
                speed = self.rng.uniform(0.5 * self.maximum_push, self.maximum_push)
                robot = self.env.scene['robot']
                velocity = robot.data.root_link_vel_w.clone()
                velocity[:, :2] += torch.tensor([np.cos(angle), np.sin(angle)],
                    device=self.env.device, dtype=velocity.dtype) * speed
                robot.write_root_link_velocity_to_sim(velocity)
                self.pushes += 1
                print(f'Random shove #{self.pushes}: {speed:.2f} m/s', flush=True)
            elif key in (ord('['), ord(']')):
                self.maximum_push = max(0.1, self.maximum_push + (0.2 if key == ord(']') else -0.2))
            elif key == ord('T'):
                self.set_mode('RECOVERY')

        robot = self.env.scene['robot']
        # Use uncorrupted physical state for switching, independent of either normalizer.
        gravity = robot.data.projected_gravity_b[0]
        self.tilt = float(torch.rad2deg(torch.acos((-gravity[2]).clamp(-1, 1))))
        self.height = float(robot.data.root_link_pos_w[0, 2])
        angular_speed = float(torch.linalg.vector_norm(robot.data.root_link_ang_vel_b[0]))
        dt = self.env.step_dt
        if self.mode == 'WALK':
            bad = self.tilt > 55 or self.height < 0.075
            self.bad_time = self.bad_time + dt if bad else 0.0
            if self.bad_time >= 0.12:
                self.set_mode('RECOVERY')
        else:
            self.recovery_time += dt
            stable = self.tilt < 20 and self.height > 0.095 and angular_speed < 1.5
            self.stable_time = self.stable_time + dt if stable else 0.0
            if self.recovery_time >= 0.8 and self.stable_time >= 0.6:
                self.set_mode('WALK')

        active_command = self.command if self.mode == 'WALK' else np.zeros(3, dtype=np.float32)
        for name, value in [('twist', active_command), ('head_pose', 0.0), ('body_pose', 0.0)]:
            self.env.command_manager.get_command(name)[:] = torch.as_tensor(value, device=self.env.device)
        actor = obs['actor'].detach().cpu().numpy().astype(np.float32, copy=True)
        actor[:, 48:51] = active_command
        actor[:, 51:61] = 0
        session = self.sessions[self.mode]
        actions = session.run(None, {session.get_inputs()[0].name: actor})[0]
        if not np.isfinite(actor).all() or not np.isfinite(actions).all():
            raise RuntimeError('Non-finite observation/action')
        self.steps += 1
        if self.steps % 50 == 0:
            Path('logs/walk_recovery_status.json').write_text(json.dumps(self.status()), encoding='utf-8')
        return torch.as_tensor(actions, device=self.env.device)


def make_env():
    cfg = load_env_cfg('Mjlab-StandUp-Flat-MicroDuck-HD1910', play=True)
    cfg.scene.num_envs = 1
    keep = {'reset_base', 'reset_robot_joints', 'reset_action_history',
            'expand_bam_friction_fields', 'set_ground_state'}
    cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
    params = cfg.events['set_ground_state'].params
    params.update(face_down_prob=0.0, face_up_prob=0.0, sitting_prob=0.0, standing_prob=1.0)
    cfg.curriculum = {}
    cfg.observations['actor'].enable_corruption = False
    for term in cfg.observations['actor'].terms.values():
        if 'max_angle_deg' in term.params:
            term.params['max_angle_deg'] = 0.0
    for term in cfg.commands.values():
        term.resampling_time_range = (10000.0, 10000.0)
    cfg.episode_length_s = 3600.0
    cfg.viewer.distance = 0.85
    return ManagerBasedRlEnv(cfg, device='cuda:0')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walk', type=Path, default=Path('models/hd1910_velocity_final.onnx'))
    parser.add_argument('--recovery', type=Path, default=Path('models/hd1910_standup_best.onnx'))
    parser.add_argument('--headless-steps', type=int, default=0)
    args = parser.parse_args()
    env = make_env()
    try:
        wrapped = RslRlVecEnvWrapper(env)
        controller = Supervisor(env, args.walk, args.recovery)

        class InteractiveViewer(NativeMujocoViewer):
            def _safe_key_callback(self, key):
                if key in [ord(c) for c in 'WSADQEXPT[]']:
                    controller.requests.put(key)
                else:
                    super()._safe_key_callback(key)

            def setup(self):
                super().setup()
                self._show_debug_vis = False
                print('VIEWER READY: W/S forward/back | A/D sideways | Q/E turn | X stop | P shove | [/] shove strength | T recover | Enter reset | Space pause', flush=True)

            def _set_status_overlay(self, viewer):
                text = (f'Controller: {controller.mode}\n'
                        f'Command vx/vy/yaw: {controller.command[0]:.2f} / {controller.command[1]:.2f} / {controller.command[2]:.2f}\n'
                        f'Tilt: {controller.tilt:.1f} deg | Height: {controller.height:.3f} m\n'
                        f'Push max: {controller.maximum_push:.1f} m/s | Pushes: {controller.pushes}\n'
                        'W/S: forward/back   A/D: sideways   Q/E: turn\n'
                        'X: stop   P: random shove   [ / ]: shove strength\n'
                        'T: recovery   Enter: reset   Space: pause\n'
                        'Commands remain active until changed. Recovery pauses motion commands.')
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value, text, ''))

        obs, _ = wrapped.reset()
        with torch.inference_mode():
            if args.headless_steps:
                for step in range(args.headless_steps):
                    if step == 50:
                        controller.requests.put(ord('W'))
                    if step == 300:
                        controller.requests.put(ord('P'))
                    if step == 500:
                        # Deterministic fallen-state integration check via the live manager.
                        event = env.event_manager.get_term_cfg('set_ground_state')
                        event.params.update(face_down_prob=1.0, standing_prob=0.0)
                        obs, _ = wrapped.reset()
                    obs, *_ = wrapped.step(controller(obs))
                Path('logs/walk_recovery_check.json').write_text(json.dumps(controller.status()), encoding='utf-8')
                print('HEADLESS RESULT:', json.dumps(controller.status()), flush=True)
                if not controller.transitions:
                    raise RuntimeError('No recovery transitions observed')
            else:
                InteractiveViewer(wrapped, controller).run()
    finally:
        env.close()


if __name__ == '__main__':
    main()
