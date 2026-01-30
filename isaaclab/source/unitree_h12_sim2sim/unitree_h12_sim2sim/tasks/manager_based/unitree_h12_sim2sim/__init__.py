# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents  # noqa: F401

##
# Register Gym environments.
##


gym.register(
    id="Unitree-H12-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_h12_sim2sim_walk_cfg:H12LocomotionFullBodyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_walk_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="Unitree-H12-Walk-RMA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_h12_walk_rma_cfg:H12LocomotionFullBodyRmaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_walk_rma_cfg:PPORunnerCfg",
    },
)
