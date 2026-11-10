"""Exercise the actual Feetech BAM control law, including episode resets."""

from dataclasses import replace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp needs CUDA")
def test_hd1910_rollout_and_partial_reset():
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-HD1910")
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=7)
        assert obs["actor"].shape == (4, 61)
        assert env.action_manager.total_action_dim == 14
        robot = env.scene["robot"]
        actuator = robot.actuators[0]
        command = actuator.get_command(robot.data)
        # A large old target must be discarded only in the world being reset.
        actuator._bam_model.actuator.q_target_smooth = command.pos.clone() + 2.0
        actuator.reset(torch.tensor([0], device=env.device))
        actuator.compute(replace(command, position_target=command.pos.clone()))
        history = actuator._bam_model.actuator.q_target_smooth
        torch.testing.assert_close(history[0], command.pos[0])
        assert torch.all(history[1:] > command.pos[1:])
        env.reset(seed=7)
        for step in range(20):
            action = torch.zeros((4, 14), device=env.device)
            action[1, 0] = 0.1
            obs, reward, _, _, _ = env.step(action)
            assert torch.isfinite(obs["actor"]).all()
            assert torch.isfinite(reward).all()
            assert torch.isfinite(env.scene["robot"].data.joint_pos).all()
            if step == 9:
                # Reset one world while the others continue their control histories.
                env.reset(env_ids=torch.tensor([0], device=env.device))
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MuJoCo Warp needs CUDA")
def test_hd1910_home_settles_upright_from_noisy_starts():
    """A fixed HOME target must hold tilt as well as height for three seconds."""
    cfg = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-HD1910")
    cfg.scene.num_envs = 8
    # Isolate nominal actuator/pose physics from robustness perturbations.
    keep = {"reset_base", "reset_robot_joints", "reset_action_history", "expand_bam_friction_fields"}
    cfg.events = {name: event for name, event in cfg.events.items() if name in keep}
    cfg.curriculum = {}
    cfg.events["reset_base"].params["pose_range"].update(
        roll=(-0.02, 0.02), pitch=(-0.02, 0.02)
    )
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.01, 0.01)
    # Do not let a fall termination reset and conceal a failed equilibrium.
    cfg.terminations = {name: term for name, term in cfg.terminations.items() if name == "time_out"}
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset(seed=13)
        actions = torch.zeros((8, 14), device=env.device)
        for _ in range(150):
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(obs["actor"]).all()
            assert torch.isfinite(reward).all()
            assert not terminated.any() and not truncated.any()
        robot = env.scene["robot"]
        # Tilt from the trunk IMU's projected gravity; actor values include noise,
        # so use the noise-free privileged observation for this physical check.
        gravity = obs["critic"][:, 6:9]
        tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1)))
        height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        print(f"HOME after 3s: tilt_deg={tilt.tolist()}, height_m={height.tolist()}")
        assert torch.all(tilt < 20), tilt
        assert torch.all(height > 0.09), height
    finally:
        env.close()
