from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)

class RmaEtSpec:
    """Specification of the privileged environment-factor vector e_t for Unitree H12 RMA.

    Reduced variant: force + leg strength + friction
    e_t = [payload_force(1), leg_strength(12), friction(1)]  -> 14 dims

    Ordering is fixed so that:
    - the env can pack e_t consistently
    - the encoder μ(e_t) can be reused across training/inference code
    - sim2sim export can carry the same latent dimension

    Indices:
    0: payload_downward_force_N
    1..12: leg_motor_strength_scale (12 values, range 0.9-1.1)
    13: ground_friction_coeff

    Notes:
    - payload should be quasi-static per episode (sample once at reset/startup).
    - strength scaling should be quasi-static per episode (0.9-1.1 range).
    - friction should be quasi-static per episode.
    """

    payload_dim: int = 1
    leg_strength_dim: int = 12
    friction_dim: int = 1

    @property
    def dim(self) -> int:
        return self.payload_dim + self.leg_strength_dim + self.friction_dim

    @property
    def payload_slice(self) -> slice:
        return slice(0, 1)

    @property
    def leg_strength_slice(self) -> slice:
        return slice(1, 1 + self.leg_strength_dim)

    @property
    def friction_slice(self) -> slice:
        start = 1 + self.leg_strength_dim
        return slice(start, start + 1)


# Default spec used by Unitree-H12-Walk-RMA-v0.
DEFAULT_ET_SPEC = RmaEtSpec()


# Leg joint ordering used for the 12-dim leg strength portion of e_t.
# This must match the description in RmaEtSpec and should remain stable across training/export.
LEG_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
