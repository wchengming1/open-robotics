"""HD1910 recovery teacher and VelStand integration, using local experts only.

Stage 1 trains the existing stand-up curriculum from random initialization.
Stage 2 warm-starts the HD1910 walking policy and distills a validated local
HD1910 stand-up expert. Do not substitute an untrained/mismatched teacher.
"""

from copy import deepcopy
from dataclasses import replace

from .microduck_hd1910_env_cfg import apply_hd1910_actuators
from .microduck_standup_env_cfg import MicroduckStandUpRlCfg, make_microduck_standup_env_cfg
from .microduck_velstand_env_cfg import MicroduckVelStandRlCfg, make_microduck_velstand_env_cfg


def make_microduck_hd1910_standup_env_cfg(play=False, rough=False):
    return apply_hd1910_actuators(make_microduck_standup_env_cfg(play=play, rough=rough))


def make_microduck_hd1910_velstand_env_cfg(play=False, rough=False):
    return apply_hd1910_actuators(make_microduck_velstand_env_cfg(play=play, rough=rough))


MicroduckHD1910StandUpRlCfg = replace(
    deepcopy(MicroduckStandUpRlCfg),
    experiment_name="microduck_hd1910_stand_teacher",
    run_name="hd1910_velstand_teacher",
    logger="tensorboard",
)

MicroduckHD1910VelStandRlCfg = replace(
    deepcopy(MicroduckVelStandRlCfg),
    experiment_name="microduck_hd1910_velstand",
    run_name="hd1910_velstand",
    logger="tensorboard",
)
MicroduckHD1910VelStandRlCfg.algorithm.bc_cfg.update(
    checkpoint_path="models/hd1910_stand_teacher.pt",
    anchor_checkpoint_path="models/hd1910_walk_teacher.pt",
    wandb_run_path=None,
    anchor_wandb_run_path=None,
)
