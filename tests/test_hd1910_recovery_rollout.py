"""Measure the HD1910 recovery robot's actual HOME equilibrium before training."""

from copy import deepcopy

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp requires CUDA")
def test_teacher_home_height_and_tilt():
    from mjlab_microduck.tasks.microduck_standup_env_cfg import STAND_Z

    cfg = load_env_cfg("Mjlab-StandUp-Flat-MicroDuck-HD1910")
    velocity = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-HD1910")
    cfg.scene.num_envs = 8
    keep = {"reset_base", "reset_robot_joints", "reset_action_history", "expand_bam_friction_fields"}
    cfg.events = {k: deepcopy(v) for k, v in velocity.events.items() if k in keep}
    cfg.events["reset_base"].params["pose_range"].update(roll=(-0.02, 0.02), pitch=(-0.02, 0.02))
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.01, 0.01)
    cfg.curriculum = {}
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset(seed=13)
        for _ in range(150):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((8, 14), device=env.device))
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(reward).all()
            assert not terminated.any() and not truncated.any()
        gravity = obs["critic"][:, 6:9]
        tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1)))
        height = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        print("Teacher HOME tilt_deg:", tilt.tolist(), "height_m:", height.tolist(), "target:", STAND_Z)
        assert torch.all(tilt < 20)
        assert torch.all((height - STAND_Z).abs() < 0.005)
    finally:
        env.close()
