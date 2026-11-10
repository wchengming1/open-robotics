"""Play a local stand-up checkpoint; press P in the viewer for a random shove."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import queue

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/hd1910_stand_teacher.pt'))
    parser.add_argument('--task', default='Mjlab-StandUp-Flat-MicroDuck-HD1910')
    parser.add_argument('--push-speed', type=float, default=0.7,
                        help='Maximum horizontal velocity impulse in m/s')
    parser.add_argument('--headless-steps', type=int, default=0)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.push_speed <= 0:
        parser.error('Checkpoint must exist and push-speed must be positive')
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    # Only manual disturbances; keep the trained robot, sensors and actuator model.
    cfg.events.pop('push_robot', None)
    cfg.curriculum = {}
    cfg.episode_length_s = 3600.0
    cfg.viewer.distance = 0.75
    agent_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=env.device)
        runner.load(str(args.checkpoint), load_cfg={'actor': True}, strict=True,
                    map_location=env.device)
        policy = runner.get_inference_policy(device=env.device)
        requests = queue.Queue()
        rng = np.random.default_rng()
        maximum = args.push_speed
        push_count = 0
        status_path = Path('logs/standup_push_status.json')

        def write_status(state):
            status_path.write_text(json.dumps(dict(status=state,
                checkpoint=str(args.checkpoint), pushes=push_count,
                push_speed=maximum)), encoding='utf-8')

        def infer(obs):
            nonlocal maximum, push_count
            while not requests.empty():
                key = requests.get_nowait()
                if key == ord('P'):
                    angle = rng.uniform(0, 2 * np.pi)
                    speed = rng.uniform(0.5 * maximum, maximum)
                    delta = torch.tensor([np.cos(angle), np.sin(angle)],
                                         device=env.device) * speed
                    robot = env.scene['robot']
                    velocity = robot.data.root_link_vel_w.clone()
                    velocity[:, :2] += delta
                    robot.write_root_link_velocity_to_sim(velocity)
                    push_count += 1
                    print(f'Push #{push_count}: delta v={delta.cpu().tolist()} m/s', flush=True)
                    write_status('running')
                else:
                    maximum = max(0.1, maximum + (0.2 if key == ord(']') else -0.2))
                    print(f'Maximum push: {maximum:.1f} m/s', flush=True)
            return policy(obs)

        class PushViewer(NativeMujocoViewer):
            def _safe_key_callback(self, key):
                # P normally toggles reward plots; reserve it for a shove here.
                if key in (ord('P'), ord('['), ord(']')):
                    requests.put(key)
                else:
                    super()._safe_key_callback(key)

            def setup(self):
                super().setup()
                write_status('running')
                print('VIEWER READY: P=random shove | [ / ]=strength | Enter=reset | Space=pause', flush=True)

            def close(self):
                super().close()
                write_status('closed')

        obs, _ = wrapped.reset()
        with torch.inference_mode():
            if args.headless_steps:
                for step in range(args.headless_steps):
                    if step == args.headless_steps // 2:
                        requests.put(ord('P'))
                    action = infer(obs)
                    if obs['actor'].shape[-1] != 61 or not torch.isfinite(action).all():
                        raise RuntimeError('Invalid policy observations/actions')
                    obs, *_ = wrapped.step(action)
                print(f'PASS: {args.headless_steps} steps, 61D observations, {push_count} random push', flush=True)
            else:
                PushViewer(wrapped, infer).run()
    finally:
        env.close()


if __name__ == '__main__':
    main()
