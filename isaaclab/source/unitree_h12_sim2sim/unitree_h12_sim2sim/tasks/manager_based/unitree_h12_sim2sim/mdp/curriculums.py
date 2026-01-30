# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from rma_modules.env_factor_spec import DEFAULT_ET_SPEC
from .rma import _apply_ground_friction, _ensure_buffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())




def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            # Increase velocity ranges if reward is high
            delta_command = torch.tensor([0.1, 0.1], device=env.device)
            # Clamp to prevent going beyond realistic velocities
            ranges.lin_vel_x = tuple(torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                -2.0,  # min velocity
                2.0,   # max velocity
            ).tolist())

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def rma_friction_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    *,
    friction_range: tuple[float, float] = (0.5, 1.0),
    num_buckets: int = 8,
    reward_term_name: str = "track_lin_vel_xy",
    min_progress: float = 0.2,
    max_progress: float = 0.8,
    advance_every_episodes: int | None = None,
    advance_every_steps: int | None = None,
    cycle: bool = True,
) -> torch.Tensor:
    """Curriculum for ground friction (global) with reward-based progression.

    The friction is global, so we track a single curriculum level and apply the
    same coefficient to all environments. Lower friction is considered harder.
    """
    if not hasattr(env, "_rma_friction_levels"):
        # Easy -> hard: high friction to low friction
        env._rma_friction_levels = torch.linspace(
            float(friction_range[1]), float(friction_range[0]), int(num_buckets), device=env.device
        )
        env._rma_friction_level = 0
        env._rma_friction_episode_count = 0

    # Advance every N steps if requested
    if advance_every_steps is not None and advance_every_steps > 0:
        if env.common_step_counter % advance_every_steps == 0:
            next_level = env._rma_friction_level + 1
            if next_level >= num_buckets:
                env._rma_friction_level = 0 if cycle else num_buckets - 1
            else:
                env._rma_friction_level = next_level
    # Otherwise update only at episode boundaries
    elif env.common_step_counter % env.max_episode_length == 0:
        env._rma_friction_episode_count += 1
        if advance_every_episodes is not None and advance_every_episodes > 0:
            if env._rma_friction_episode_count % advance_every_episodes == 0:
                next_level = env._rma_friction_level + 1
                if next_level >= num_buckets:
                    env._rma_friction_level = 0 if cycle else num_buckets - 1
                else:
                    env._rma_friction_level = next_level
        else:
            reward_term = env.reward_manager.get_term_cfg(reward_term_name)
            reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

            if reward > reward_term.weight * max_progress:
                env._rma_friction_level = min(env._rma_friction_level + 1, num_buckets - 1)
            elif reward < reward_term.weight * min_progress:
                env._rma_friction_level = max(env._rma_friction_level - 1, 0)

    friction_val = float(env._rma_friction_levels[env._rma_friction_level].item())
    env._rma_curriculum_friction = friction_val

    # Update e_t buffer (global friction shared across envs)
    et = _ensure_buffer(env, "rma_env_factors_buf", DEFAULT_ET_SPEC.dim)
    et[:, DEFAULT_ET_SPEC.friction_slice] = friction_val

    # Apply to simulator (global terrain material)
    _apply_ground_friction(env, friction_val)

    return env._rma_friction_levels[env._rma_friction_level]