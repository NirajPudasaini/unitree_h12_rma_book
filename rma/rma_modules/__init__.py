"""Reusable neural modules for RMA (Rapid Motor Adaptation).

Phase 1 (privileged):
- `EnvFactorEncoder` maps sim-only env factors e_t -> z_t.
- `EnvFactorDecoder` reconstructs e_t from z_t (for debugging/supervision).

Phase 2 (adaptation):
- `AdaptationModule` maps history of observations/actions -> \hat{z}_t.
"""

from .env_factor_encoder import EnvFactorEncoder, EnvFactorEncoderCfg
from .env_factor_decoder import EnvFactorDecoder, EnvFactorDecoderCfg
from .adaptation_module import AdaptationModule
from .env_factor_spec import DEFAULT_ET_SPEC, RmaEtSpec

__all__ = [
    "EnvFactorEncoder",
    "EnvFactorEncoderCfg",
    "EnvFactorDecoder",
    "EnvFactorDecoderCfg",
    "AdaptationModule",
    "RmaEtSpec",
    "DEFAULT_ET_SPEC",
]
