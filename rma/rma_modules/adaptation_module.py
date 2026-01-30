from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .env_factor_encoder import _build_mlp


@dataclass
class AdaptationModuleCfg:
    """Simple feed-forward adaptation module config.

    This is a scaffold for phase-2 (online adaptation). A common choice is to feed a short history window of
    proprioception + actions (flattened) and output an estimate of z_t.
    """

    in_dim: int
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"


class AdaptationModule(nn.Module):
    """Maps observation/action history -> \hat{z}_t."""

    def __init__(self, cfg: AdaptationModuleCfg):
        super().__init__()
        self.cfg = cfg
        self.net = _build_mlp(
            in_dim=self.cfg.in_dim,
            hidden_dims=list(self.cfg.hidden_dims),
            out_dim=self.cfg.latent_dim,
            activation=self.cfg.activation,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 2 or history.shape[-1] != self.cfg.in_dim:
            raise ValueError(f"history must be (N, {self.cfg.in_dim}); got {tuple(history.shape)}")
        return self.net(history)
