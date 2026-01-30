#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

"""Simple script to verify decoder reconstruction accuracy - no IsaacLab required."""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Direct imports - read the actual implementation files
# to avoid full IsaacLab initialization
import importlib.util

def load_rma_module(module_name, file_path):
    """Load a module from file path without full environment."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load the actual implementations from source files
rma_encoder_module = load_rma_module(
    "env_factor_encoder",
    "/home/niraj/isaac_projects/unitree_h12_rma/isaaclab/source/unitree_h12_sim2sim/unitree_h12_sim2sim/rma_modules/env_factor_encoder.py"
)
rma_decoder_module = load_rma_module(
    "env_factor_decoder", 
    "/home/niraj/isaac_projects/unitree_h12_rma/isaaclab/source/unitree_h12_sim2sim/unitree_h12_sim2sim/rma_modules/env_factor_decoder.py"
)

EnvFactorEncoder = rma_encoder_module.EnvFactorEncoder
EnvFactorEncoderCfg = rma_encoder_module.EnvFactorEncoderCfg
EnvFactorDecoder = rma_decoder_module.EnvFactorDecoder
EnvFactorDecoderCfg = rma_decoder_module.EnvFactorDecoderCfg


class EnvFactorNormalizer:
    """Normalizes environment factors to [0, 1] range."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.mins = torch.tensor([
            0.0,      # payload force min
            0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,  # leg strengths min
            0.0,      # friction min
            0.0,      # terrain amplitude min
            0.0,      # terrain lengthscale min
            0.0,      # terrain noise min
        ], device=device, dtype=torch.float32)
        
        self.maxs = torch.tensor([
            50.0,     # payload force max
            1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,  # leg strengths max
            1.0,      # friction max
            0.002,    # terrain amplitude max
            0.2,      # terrain lengthscale max
            0.1,      # terrain noise max
        ], device=device, dtype=torch.float32)
        
        self.ranges = self.maxs - self.mins
    
    def normalize(self, e_t: torch.Tensor) -> torch.Tensor:
        return (e_t - self.mins) / (self.ranges + 1e-8)
    
    def denormalize(self, e_t_normalized: torch.Tensor) -> torch.Tensor:
        ranges = self.ranges.to(e_t_normalized.device)
        mins = self.mins.to(e_t_normalized.device)
        return e_t_normalized * ranges + mins


def find_latest_checkpoint(log_dir: str, model_type: str = "encoder") -> str:
    """Find the latest checkpoint - prioritizes _latest.pt files."""
    checkpoint_dir = Path(log_dir) / "checkpoints" / model_type
    
    if not checkpoint_dir.exists():
        return None
    
    # First, try to find the _latest.pt file (most up-to-date)
    latest = checkpoint_dir / f"{model_type}_latest.pt"
    if latest.exists():
        return str(latest)
    
    # Fallback: find the highest iteration checkpoint
    checkpoints = list(checkpoint_dir.glob(f"{model_type}_iter_*.pt"))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
        return str(checkpoints[-1])
    
    return None


def load_models(encoder_path: str, decoder_path: str, device: str = "cuda"):
    """Load encoder and decoder models."""
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_path}")
    
    # Load encoder
    encoder_ckpt = torch.load(encoder_path, map_location=device)
    encoder_cfg = EnvFactorEncoderCfg(in_dim=17, latent_dim=8, hidden_dims=(256, 128))
    encoder = EnvFactorEncoder(cfg=encoder_cfg).to(device)
    encoder.load_state_dict(encoder_ckpt["model_state_dict"], strict=False)
    encoder.eval()
    
    # Load decoder
    decoder_ckpt = torch.load(decoder_path, map_location=device)
    decoder_cfg = EnvFactorDecoderCfg()
    decoder = EnvFactorDecoder(cfg=decoder_cfg).to(device)
    # Filter out non-module keys like _output_ranges
    state_dict = {k: v for k, v in decoder_ckpt["model_state_dict"].items() if not k.startswith("_")}
    decoder.load_state_dict(state_dict, strict=False)
    decoder.eval()
    
    return encoder, decoder, encoder_cfg, decoder_cfg


def compute_reconstruction_stats(encoder, decoder, e_t, device: str = "cuda") -> dict:
    """Compute reconstruction statistics with normalization."""
    e_t = e_t.to(device)
    
    # Normalize inputs for encoder/decoder
    normalizer = EnvFactorNormalizer(device=device)
    e_t_normalized = normalizer.normalize(e_t)
    
    with torch.no_grad():
        z_t = encoder(e_t_normalized)
        e_t_recon_normalized = decoder(z_t, apply_scaling=False)  # No scaling when normalized
        # Denormalize output back to original range
        e_t_recon = normalizer.denormalize(e_t_recon_normalized)
    
    mse = torch.nn.functional.mse_loss(e_t, e_t_recon, reduction="none")
    mae = torch.nn.functional.l1_loss(e_t, e_t_recon, reduction="none")
    
    factor_names = [
        "Payload Force (N)",
        *[f"Leg Strength {i}" for i in range(12)],
        "Friction",
        "Terrain Amplitude (m)",
        "Terrain Lengthscale (m)",
        "Terrain Noise Step",
    ]
    
    stats = {
        "total_mse": mse.mean().item(),
        "total_mae": mae.mean().item(),
        "total_rmse": mse.mean().sqrt().item(),
        "per_factor": {}
    }
    
    for i, name in enumerate(factor_names):
        stats["per_factor"][name] = {
            "mse": mse[:, i].mean().item(),
            "mae": mae[:, i].mean().item(),
            "rmse": mse[:, i].mean().sqrt().item(),
            "min": e_t[:, i].min().item(),
            "max": e_t[:, i].max().item(),
            "mean": e_t[:, i].mean().item(),
            "std": e_t[:, i].std().item(),
        }
    
    return stats


def plot_reconstruction_comparison(encoder, decoder, e_t, output_dir: str = "results"):
    """Plot actual vs predicted for each factor with normalization."""
    e_t_orig = e_t.cpu()
    device = next(encoder.parameters()).device
    
    # Normalize, encode, decode, denormalize
    normalizer = EnvFactorNormalizer(device=device)
    e_t_normalized = normalizer.normalize(e_t.to(device))
    
    with torch.no_grad():
        z_t = encoder(e_t_normalized)
        e_t_recon_normalized = decoder(z_t, apply_scaling=False)
        e_t_recon = normalizer.denormalize(e_t_recon_normalized).cpu()
    
    factor_names = [
        "Payload Force (N)",
        *[f"Leg Strength {i}" for i in range(12)],
        "Friction",
        "Terrain Amplitude (m)",
        "Terrain Lengthscale (m)",
        "Terrain Noise Step",
    ]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a grid of subplots (17 factors)
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for i, name in enumerate(factor_names):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        actual = e_t_orig[:, i].numpy()
        predicted = e_t_recon[:, i].numpy()
        
        # Scatter plot with diagonal line
        ax.scatter(actual, predicted, alpha=0.5, s=10)
        
        # Add perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('Actual', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{name}\nMSE={np.mean((actual-predicted)**2):.6f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = Path(output_dir) / "actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Saved plot to: {plot_path}")
    
    # Also create error distribution plots
    fig2, axes = plt.subplots(6, 3, figsize=(20, 24))
    axes = axes.flatten()
    
    for i, name in enumerate(factor_names):
        ax = axes[i]
        actual = e_t[:, i].numpy()
        predicted = e_t_recon[:, i].numpy()
        error = predicted - actual
        
        ax.hist(error, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name}\nMean Error={np.mean(error):.6f}, Std={np.std(error):.6f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_plot_path = Path(output_dir) / "error_distribution.png"
    plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved error distribution plot to: {error_plot_path}")
    
    plt.close('all')


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Verify decoder reconstruction accuracy")
    parser.add_argument("--log-dir", type=str, default=None, help="Path to training log directory")
    parser.add_argument("--encoder", type=str, default=None, help="Path to encoder checkpoint")
    parser.add_argument("--decoder", type=str, default=None, help="Path to decoder checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num-samples", type=int, default=1024, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    encoder_path = args.encoder
    decoder_path = args.decoder
    
    if args.log_dir:
        print(f"[INFO] Using log directory: {args.log_dir}")
        if not encoder_path:
            encoder_path = find_latest_checkpoint(args.log_dir, "encoder")
        if not decoder_path:
            decoder_path = find_latest_checkpoint(args.log_dir, "decoder")
    
    if not encoder_path or not decoder_path:
        print("[INFO] Searching for latest training run...")
        log_root = Path("/home/niraj/isaac_projects/unitree_h12_rma/wbc_agile_utils/logs/rsl_rl/unitree_h12_walk_rma")
        if log_root.exists():
            runs = sorted(log_root.glob("2026-*"))
            if runs:
                latest_run = runs[-1]
                print(f"[INFO] Found latest run: {latest_run.name}")
                if not encoder_path:
                    encoder_path = find_latest_checkpoint(str(latest_run), "encoder")
                if not decoder_path:
                    decoder_path = find_latest_checkpoint(str(latest_run), "decoder")
    
    if not encoder_path or not decoder_path:
        print("[ERROR] Could not find encoder/decoder checkpoints")
        print("[ERROR] Note: Training may still be saving checkpoints")
        print(f"[ERROR] Checked: {log_root if 'log_root' in locals() else 'N/A'}")
        return
    
    print(f"\n[INFO] Encoder checkpoint: {encoder_path}")
    print(f"[INFO] Decoder checkpoint: {decoder_path}")
    
    # Load models
    print("\n[INFO] Loading models...")
    try:
        encoder, decoder, encoder_cfg, decoder_cfg = load_models(
            encoder_path, decoder_path, device=args.device
        )
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"[INFO] Encoder: in_dim={encoder_cfg.in_dim}, latent_dim={encoder_cfg.latent_dim}")
    print(f"[INFO] Decoder: output_dim=17")
    
    # Generate test samples
    print(f"\n[INFO] Generating {args.num_samples} test samples...")
    e_t_test = torch.randn(args.num_samples, 17)
    
    # Scale to realistic ranges
    e_t_test[:, 0] = torch.clamp(e_t_test[:, 0] * 12.5 + 25, 0, 50)      # Force
    e_t_test[:, 1:13] = torch.clamp(e_t_test[:, 1:13] * 0.05 + 1.0, 0.9, 1.1)
    e_t_test[:, 13] = torch.clamp(e_t_test[:, 13] * 0.3 + 0.5, 0, 1)
    e_t_test[:, 14] = torch.clamp(e_t_test[:, 14] * 0.5e-3, 0, 2e-3)
    e_t_test[:, 15] = torch.clamp(e_t_test[:, 15] * 0.05 + 0.1, 0, 0.2)
    e_t_test[:, 16] = torch.clamp(e_t_test[:, 16] * 0.025 + 0.05, 0, 0.1)
    
    # Compute reconstruction statistics
    print("\n[INFO] Computing reconstruction statistics...")
    stats = compute_reconstruction_stats(encoder, decoder, e_t_test, device=args.device)
    
    # Generate plots
    print("\n[INFO] Generating visualizations...")
    output_dir = Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_reconstruction_comparison(encoder, decoder, e_t_test, output_dir=str(output_dir))
    
    # Print results
    print("\n" + "="*100)
    print("DECODER RECONSTRUCTION ACCURACY REPORT")
    print("="*100)
    print(f"Test samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("-"*100)
    
    print(f"\nOverall Reconstruction Error:")
    print(f"  Mean Squared Error (MSE):       {stats['total_mse']:.6f}")
    print(f"  Mean Absolute Error (MAE):     {stats['total_mae']:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {stats['total_rmse']:.6f}")
    
    print(f"\nPer-Factor Reconstruction Error:")
    print("-"*100)
    print(f"{'Factor':<30} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Range':<20}")
    print("-"*100)
    
    for factor_name, factor_stats in stats["per_factor"].items():
        range_str = f"[{factor_stats['min']:.4f}, {factor_stats['max']:.4f}]"
        print(
            f"{factor_name:<30} "
            f"{factor_stats['mse']:<12.6f} "
            f"{factor_stats['mae']:<12.6f} "
            f"{factor_stats['rmse']:<12.6f} "
            f"{range_str:<20}"
        )
    
    print("\n" + "="*100)
    
    # Interpretation
    print("\nInterpretation Guide:")
    print("  MSE < 0.001: Excellent reconstruction (< 0.1% error)")
    print("  MSE < 0.01:  Good reconstruction (< 1% error)")
    print("  MSE < 0.1:   Acceptable reconstruction (< 10% error)")
    print("  MSE > 0.1:   Poor reconstruction (> 10% error)")
    
    # Quality assessment
    print("\nQuality Assessment:")
    if stats['total_mse'] < 0.001:
        print("  ✅ EXCELLENT: Decoder provides excellent reconstruction")
    elif stats['total_mse'] < 0.01:
        print("  ✅ GOOD: Decoder provides good reconstruction")
    elif stats['total_mse'] < 0.1:
        print("  ⚠️  ACCEPTABLE: Decoder reconstruction is acceptable but could improve")
    else:
        print("  ❌ POOR: Decoder reconstruction needs improvement")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    main()
