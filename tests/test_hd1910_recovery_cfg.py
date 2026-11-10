"""HD1910 recovery must preserve each base task's model and stay offline."""

import mujoco
import pytest
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


@pytest.mark.parametrize("family", ["StandUp", "VelStand"])
@pytest.mark.parametrize("surface", ["Flat", "Rough"])
@pytest.mark.parametrize("play", [False, True])
def test_hd1910_recovery_keeps_base_physics_and_contract(family, surface, play):
    base_id = f"Mjlab-{family}-{surface}-MicroDuck"
    cfg = load_env_cfg(base_id + "-HD1910", play=play)
    original = load_env_cfg(base_id, play=play)
    robot = cfg.scene.entities["robot"]
    actuator = robot.articulation.actuators[0]
    assert type(actuator).__name__ == "FeetechBamActuatorCfg"
    assert actuator.json_path.endswith("1910_m6.json")
    assert actuator.kp_fw == 5.0
    assert actuator.vin_range == (7.4, 8.0)
    assert actuator.vin_min == 7.0
    assert robot.spec_fn == original.scene.entities["robot"].spec_fn
    assert robot.init_state == original.scene.entities["robot"].init_state
    assert list(cfg.observations["actor"].terms) == list(original.observations["actor"].terms)
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(0.02)
    assert "expand_bam_friction_fields" in cfg.events
    model = robot.spec_fn().compile()
    names = [model.joint(i).name for i in range(model.njnt)
             if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
             and not model.joint(i).name.startswith("passive_")]
    assert len(names) == 14 and model.nu == 14
    assert original.scene.entities["robot"].articulation.actuators[0].motor_name == "xl330"
    assert load_rl_cfg(base_id + "-HD1910").experiment_name != load_rl_cfg(base_id).experiment_name


def test_velstand_requires_local_hd1910_teachers_without_changing_original():
    cfg = load_rl_cfg("Mjlab-VelStand-Flat-MicroDuck-HD1910")
    bc = cfg.algorithm.bc_cfg
    assert bc["checkpoint_path"] == "models/hd1910_stand_teacher.pt"
    assert bc["anchor_checkpoint_path"] == "models/hd1910_walk_teacher.pt"
    assert bc["wandb_run_path"] is None and bc["anchor_wandb_run_path"] is None
    assert bc["anchor_coef"] > 0
    assert cfg.actor.obs_normalization and cfg.critic.obs_normalization
    original = load_rl_cfg("Mjlab-VelStand-Flat-MicroDuck")
    assert original.algorithm.bc_cfg["checkpoint_path"] is None
    assert original.algorithm.bc_cfg["wandb_run_path"] is not None
