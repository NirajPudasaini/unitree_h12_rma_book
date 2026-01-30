"""Environment factor decoder for reconstructing e_t from latent z_t or observations.

This module provides the inverse of the env_factor_encoder: given a latent z_t (or observations),
it reconstructs/predicts the environment factors e_t. Can be trained with supervision from actual
e_t data collected during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    """Build a multi-layer perceptron with specified architecture."""
    if activation == "elu":
        act: type[nn.Module] = nn.ELU
    elif activation == "relu":
        act = nn.ReLU
    elif activation == "tanh":
        act = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    layers: list[nn.Module] = []
    last_dim = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(act())
        last_dim = h
    layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)


@dataclass
class EnvFactorDecoderCfg:
    """Configuration for environment factor decoder.
    
    The decoder can operate in two modes:
    1. Latent mode: input is z_t (latent from encoder), reconstruct e_t
    2. Observation mode: input is observation features, predict e_t
    """
    
    in_dim: int = 8  # Default: latent_dim from encoder
    out_dim: int = 14  # e_t dimension (1 + 12 + 1)
    hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "elu"
    use_output_scaling: bool = True  # Scale outputs to e_t ranges
    
    # Output ranges for each factor (used for post-processing)
    payload_range: tuple[float, float] = (0.0, 50.0)  # Force: 0-50 N
    leg_strength_range: tuple[float, float] = (0.9, 1.1)  # Strength: 0.9-1.1
    friction_range: tuple[float, float] = (0.5, 2.0)  # Friction: reasonable range
    terrain_amplitude_range: tuple[float, float] = (0.0, 0.01)  # Amplitude: 0-10mm
    terrain_lengthscale_range: tuple[float, float] = (0.05, 0.2)  # Lengthscale: 5-20cm
    terrain_noise_step_range: tuple[float, float] = (0.01, 0.1)  # Noise step: 1-10cm


class EnvFactorDecoder(nn.Module):
    """Decodes latent z_t or observations back into environment factors e_t.
    
    Can be trained with MSE loss using actual e_t data as supervision:
        loss = MSE(decoder(z_t), e_t_actual)
    
    This provides a way to:
    1. Debug and understand what the encoder is learning
    2. Reconstruct e_t from latent for analysis
    3. Train a secondary model that predicts e_t from observations (model-based adaptation)
    """

    def __init__(self, cfg: EnvFactorDecoderCfg | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvFactorDecoderCfg()
        
        self.net = _build_mlp(
            in_dim=self.cfg.in_dim,
            hidden_dims=list(self.cfg.hidden_dims),
            out_dim=self.cfg.out_dim,
            activation=self.cfg.activation,
        )
        
        if self.cfg.use_output_scaling:
            # Register output scaling parameters (not trainable)
            self._register_output_ranges()
    
    def _register_output_ranges(self) -> None:
        """Register output ranges for post-processing."""
        # e_t structure: [force(1), leg_strength(12), friction(1)]
        ranges = []
        # Payload force: 1 dim, range 0-50 N
        ranges.append(self.cfg.payload_range)
        # Leg strength: 12 dims, each 0.9-1.1
        for _ in range(12):
            ranges.append(self.cfg.leg_strength_range)
        # Friction: 1 dim
        ranges.append(self.cfg.friction_range)
        # Convert to tensor for batch processing
        ranges_tensor = torch.tensor(ranges, dtype=torch.float32)  # (out_dim, 2)
        self.register_buffer("_output_ranges", ranges_tensor)
    
    def forward(self, latent: torch.Tensor, apply_scaling: bool = True) -> torch.Tensor:
        """Decode latent z_t (or observations) to environment factors e_t.
        
        Args:
            latent: Tensor of shape (N, in_dim), typically z_t from encoder
            apply_scaling: If True, scale outputs to valid e_t ranges
        
        Returns:
            e_t: Tensor of shape (N, out_dim=17) with decoded environment factors
        """
        if latent.ndim != 2 or latent.shape[-1] != self.cfg.in_dim:
            raise ValueError(
                f"latent must be (N, {self.cfg.in_dim}); got {tuple(latent.shape)}"
            )
        
        # Forward through MLP (raw output, usually unbounded)
        e_t_raw = self.net(latent)
        
        if apply_scaling and self.cfg.use_output_scaling:
            e_t_raw = self._apply_output_scaling(e_t_raw)
        
        return e_t_raw
    
    def _apply_output_scaling(self, e_t_raw: torch.Tensor) -> torch.Tensor:
        """Scale raw network outputs to valid e_t ranges using sigmoid/tanh.
        
        Args:
            e_t_raw: Raw MLP output (N, out_dim)
        
        Returns:
            Scaled e_t (N, out_dim) within valid ranges
        """
        device = e_t_raw.device
        ranges = self._output_ranges.to(device)  # (out_dim, 2)
        
        # Normalize: scale using sigmoid to [0, 1], then map to [min, max]
        e_t_normalized = torch.sigmoid(e_t_raw)  # (N, out_dim) in [0, 1]
        
        min_vals = ranges[:, 0]  # (out_dim,)
        max_vals = ranges[:, 1]  # (out_dim,)
        
        # Scale from [0, 1] to [min, max]
        e_t_scaled = min_vals + e_t_normalized * (max_vals - min_vals)
        
        return e_t_scaled
    
    def compute_reconstruction_loss(
        self,
        latent: torch.Tensor,
        e_t_target: torch.Tensor,
        loss_fn: nn.Module | None = None,
        apply_scaling: bool = True,
    ) -> torch.Tensor:
        """Compute supervised reconstruction loss.
        
        Args:
            latent: Latent encoding (N, in_dim)
            e_t_target: Ground-truth environment factors (N, out_dim)
            loss_fn: Loss function (default: MSE)
            apply_scaling: Whether to apply output scaling
        
        Returns:
            Scalar loss value
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        e_t_pred = self.forward(latent, apply_scaling=apply_scaling)
        loss = loss_fn(e_t_pred, e_t_target)
        return loss
    
    def get_factor_predictions(self, latent: torch.Tensor, apply_scaling: bool = True) -> dict[str, torch.Tensor]:
        """Decode and return individual environment factors.
        Args:
            latent: Latent encoding (N, in_dim)
            apply_scaling: Whether to apply output scaling
        Returns:
            Dictionary with keys: 'payload_force', 'leg_strength', 'friction'
        """
        e_t = self.forward(latent, apply_scaling=apply_scaling)
        return {
            "payload_force": e_t[:, 0:1],  # (N, 1)
            "leg_strength": e_t[:, 1:13],  # (N, 12)
            "friction": e_t[:, 13:14],  # (N, 1)
        }
