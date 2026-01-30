# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RMA-ready variant of the Unitree H12 walking task.

Phase-1 (privileged) training intent:
- The policy consumes an *extrinsics* vector z_t (latent) appended to the usual proprioceptive observation.
- The environment can expose a separate *environment factor* vector e_t (privileged, sim-only) that will later be
  encoded into z_t by the env-factor encoder (implemented under unitree_h12_sim2sim.rma_modules).

This file only changes observation wiring and keeps dynamics identical to Unitree-H12-Walk-v0.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .unitree_h12_sim2sim_walk_cfg import (
    EventCfg as _BaseEventCfg,
    H12LocomotionFullBodyEnvCfg,
    ObservationsCfg as _BaseObservationsCfg,
)

from rma_modules.env_factor_spec import DEFAULT_ET_SPEC


RMA_Z_DIM = 8
# e_t now only includes: force, leg_strength (no terrain)
RMA_ET_DIM = DEFAULT_ET_SPEC.dim



# Reformulated for base policy and RMA encoder/decoder training (no terrain in e_t)
@configclass
class ObservationsRmaCfg(_BaseObservationsCfg):
    """Observation specification for RMA: extrinsics (z_t) and reduced env factors (e_t: force, leg_strength)."""

    @configclass
    class PolicyCfg(_BaseObservationsCfg.PolicyCfg):
        """Observations for policy group (base policy or RMA policy)."""

        # RMA latent (z_t) appended to the end of the policy observation.
        # For base policy training, this can be zeros or omitted; for RMA, populated by encoder.
        rma_extrinsics = ObsTerm(func=mdp.rma_extrinsics, params={"dim": RMA_Z_DIM})

        def __post_init__(self):
            super().__post_init__()
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(_BaseObservationsCfg.CriticCfg):
        """Observations for critic group (privileged, for RMA phase-1 or logging)."""

        # Privileged environment factors (e_t): force, leg_strength only.
        rma_env_factors = ObsTerm(func=mdp.rma_env_factors, params={"dim": RMA_ET_DIM})
        # Slip-related cue: tangential foot velocity during contact (privileged).
        feet_slip_velocity = ObsTerm(
            func=mdp.feet_slip_velocity,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            },
        )

        def __post_init__(self):
            super().__post_init__()

    critic: CriticCfg = CriticCfg()

@configclass
class CurriculumRmaCfg:
    """Curriculum config for RMA training.

    Enable terrain curriculum so terrain changes are driven by curriculum.
    Keep command curriculum enabled.
    """

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
@configclass
class H12LocomotionFullBodyRmaEnvCfg(H12LocomotionFullBodyEnvCfg):
    """Configuration for H12 full-body locomotion task with RMA observation extensions."""

    observations: ObservationsRmaCfg = ObservationsRmaCfg()
    curriculum: CurriculumRmaCfg = CurriculumRmaCfg()

    # Add EventTerm to sample rma_env_factors at reset
    @configclass
    class EventsCfg(_BaseEventCfg):
        """Events for RMA: sample rma_env_factors at reset."""
        sample_rma_env_factors = EventTerm(
            func=mdp.sample_rma_env_factors,
            mode="reset",  # call at every environment reset
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    events: EventsCfg = EventsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Ensure terrain generator curriculum is enabled for RMA training.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
