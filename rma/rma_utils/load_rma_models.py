# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility script to load and inspect RMA models (encoder, decoder, policy)."""

import argparse
import os
import torch
from pathlib import Path

from rma_modules import (
    EnvFactorEncoder,
    EnvFactorEncoderCfg,
    EnvFactorDecoder,
    EnvFactorDecoderCfg,
)


def load_encoder(checkpoint_path: str, device: str = "cuda") -> tuple[EnvFactorEncoder, dict]:
    """Load encoder model and config from checkpoint.

    Args:
        checkpoint_path: Path to encoder checkpoint file.
        device: Device to load model to.

    Returns:
        Tuple of (encoder_model, config_dict).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create encoder with default config
    encoder_cfg = EnvFactorEncoderCfg(in_dim=17, latent_dim=8, hidden_dims=(256, 128))
    encoder = EnvFactorEncoder(cfg=encoder_cfg)
    
    # Load weights
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder = encoder.to(device)
    encoder.eval()
    
    config = checkpoint.get("config", {})
    return encoder, config


def load_decoder(checkpoint_path: str, device: str = "cuda") -> tuple[EnvFactorDecoder, dict]:
    """Load decoder model and config from checkpoint.

    Args:
        checkpoint_path: Path to decoder checkpoint file.
        device: Device to load model to.

    Returns:
        Tuple of (decoder_model, config_dict).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create decoder with default config
    decoder_cfg = EnvFactorDecoderCfg()
    decoder = EnvFactorDecoder(cfg=decoder_cfg)
    
    # Load weights
    decoder.load_state_dict(checkpoint["model_state_dict"])
    decoder = decoder.to(device)
    decoder.eval()
    
    config = checkpoint.get("config", {})
    return decoder, config


def test_encoder_decoder(encoder: EnvFactorEncoder, decoder: EnvFactorDecoder, device: str = "cuda"):
    """Test encoder and decoder with random input.

    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        device: Device to run on.
    """
    print("\n[INFO] Testing Encoder/Decoder Forward Pass")
    print("=" * 80)
    
    # Create random environment factors
    batch_size = 4
    e_t = torch.randn(batch_size, 17, device=device)
    
    print(f"Input e_t shape: {e_t.shape}")
    print(f"  - Payload force [0]: {e_t[:, 0].mean().item():.4f} +/- {e_t[:, 0].std().item():.4f}")
    print(f"  - Leg strength [1:13] mean: {e_t[:, 1:13].mean().item():.4f}")
    print(f"  - Friction [13]: {e_t[:, 13].mean().item():.4f}")
    print(f"  - Terrain [14:17] mean: {e_t[:, 14:17].mean().item():.4f}")
    
    # Encode
    with torch.no_grad():
        z_t = encoder(e_t)
    print(f"\nEncoder output z_t shape: {z_t.shape}")
    print(f"  Latent values: {z_t[0].detach().cpu().numpy()}")
    
    # Decode
    with torch.no_grad():
        e_t_reconstructed = decoder(z_t, apply_scaling=True)
    print(f"\nDecoder output ê_t shape: {e_t_reconstructed.shape}")
    
    # Compute reconstruction error
    with torch.no_grad():
        mse_loss = torch.nn.functional.mse_loss(e_t, e_t_reconstructed)
    print(f"\nReconstruction MSE Loss: {mse_loss.item():.6f}")
    
    # Per-factor reconstruction error
    with torch.no_grad():
        per_factor_mse = torch.nn.functional.mse_loss(e_t, e_t_reconstructed, reduction="none").mean(dim=0)
    
    print("\nPer-factor reconstruction error:")
    factor_names = [
        "Payload Force",
        *[f"Leg Strength {i}" for i in range(12)],
        "Friction",
        "Terrain Amplitude",
        "Terrain Lengthscale",
        "Terrain Noise Step",
    ]
    
    for i, (name, mse) in enumerate(zip(factor_names, per_factor_mse)):
        print(f"  [{i:2d}] {name:25s}: {mse.item():.6f}")


def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint contents.

    Args:
        checkpoint_path: Path to checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] File not found: {checkpoint_path}")
        return
    
    print(f"\n[INFO] Inspecting checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"\nModel state dict layers:")
        total_params = 0
        for name, param in state_dict.items():
            params = param.numel()
            total_params += params
            print(f"  {name:40s}: {tuple(param.shape)} ({params:,} params)")
        print(f"\nTotal parameters: {total_params:,}")
    
    if "iteration" in checkpoint:
        print(f"\nIteration: {checkpoint['iteration']}")
    
    if "config" in checkpoint:
        print(f"\nConfig: {checkpoint['config']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Load and inspect RMA models.")
    parser.add_argument("--encoder", type=str, default=None, help="Path to encoder checkpoint.")
    parser.add_argument("--decoder", type=str, default=None, help="Path to decoder checkpoint.")
    parser.add_argument("--test", action="store_true", help="Test forward pass through encoder/decoder.")
    parser.add_argument("--inspect", type=str, default=None, help="Inspect checkpoint contents.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load models to.")
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_checkpoint(args.inspect)
        return
    
    encoder = None
    decoder = None
    
    if args.encoder:
        print(f"[INFO] Loading encoder from: {args.encoder}")
        encoder, enc_cfg = load_encoder(args.encoder, device=args.device)
        print(f"[INFO] Encoder loaded successfully")
        print(f"       Config: {enc_cfg}")
    
    if args.decoder:
        print(f"[INFO] Loading decoder from: {args.decoder}")
        decoder, dec_cfg = load_decoder(args.decoder, device=args.device)
        print(f"[INFO] Decoder loaded successfully")
        print(f"       Config: {dec_cfg}")
    
    if args.test and encoder and decoder:
        test_encoder_decoder(encoder, decoder, device=args.device)
    elif args.test and not (encoder and decoder):
        print("[ERROR] Both encoder and decoder required for testing. Use --encoder and --decoder.")


if __name__ == "__main__":
    main()
