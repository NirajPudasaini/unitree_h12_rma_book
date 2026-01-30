#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deploy RMA policy with adaptation module (Phase 2 student deployment).

This script evaluates the trained policy using the adaptation module to predict
z_t from observation/action history, WITHOUT access to privileged environment
factors (e_t). This simulates real-world deployment.

Usage:
    python rma/deploy_with_adaptation.py \
        --task Unitree-H12-Walk-RMA-v0 \
        --policy_checkpoint logs/rsl_rl/.../model_*.pt \
        --adaptation_checkpoint rma/outputs/adaptation_dataset/.../adaptation_model/adaptation_latest.pt \
        --num_envs 16 \
        --num_steps 2000 \
        --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

# Add local packages to PYTHONPATH
workspace_root = Path(__file__).resolve().parents[1]
isaaclab_root = Path("/home/niraj/isaac_projects/IsaacLab")
sys.path.insert(0, str(isaaclab_root / "source"))
sys.path.insert(0, str(workspace_root / "isaaclab" / "source"))
sys.path.insert(0, str(workspace_root / "rma"))
sys.path.insert(0, str(workspace_root / "isaaclab" / "scripts" / "rsl_rl"))

from rma_modules import AdaptationModule, AdaptationModuleCfg


@dataclass
class EvalMetrics:
    """Evaluation metrics for deployment."""
    total_steps: int = 0
    total_episodes: int = 0
    total_reward: float = 0.0
    episode_rewards: list = None
    episode_lengths: list = None
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_lengths is None:
            self.episode_lengths = []
    
    @property
    def mean_reward(self) -> float:
        return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
    
    @property
    def mean_length(self) -> float:
        return np.mean(self.episode_lengths) if self.episode_lengths else 0.0
    
    def summary(self) -> str:
        return (
            f"Episodes: {self.total_episodes}\n"
            f"Mean Episode Reward: {self.mean_reward:.2f} ± {np.std(self.episode_rewards):.2f}\n"
            f"Mean Episode Length: {self.mean_length:.1f} ± {np.std(self.episode_lengths):.1f}\n"
            f"Total Steps: {self.total_steps}"
        )


def _get_unwrapped(env):
    """Get the innermost unwrapped environment."""
    while hasattr(env, "env"):
        env = env.env
    return env


def _maybe_tuple_obs(obs):
    """Handle tuple observations from reset."""
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def _step_env(env, action):
    """Step environment, handle both 4 and 5 return value formats."""
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = terminated | truncated
        return obs, reward, done, info
    obs, reward, done, info = step_out
    return obs, reward, done, info


def _write_extrinsics(env, z_t: torch.Tensor) -> None:
    """Write z_t into env.rma_extrinsics_buf."""
    unwrapped = _get_unwrapped(env)
    z_buf = getattr(unwrapped, "rma_extrinsics_buf", None)
    if z_buf is None or z_buf.shape != (unwrapped.num_envs, z_t.shape[-1]):
        z_buf = torch.zeros(
            (unwrapped.num_envs, z_t.shape[-1]), 
            device=unwrapped.device, 
            dtype=torch.float
        )
        setattr(unwrapped, "rma_extrinsics_buf", z_buf)
    z_buf.copy_(z_t.to(unwrapped.device))


def _load_policy(checkpoint_path: str, env, agent_cfg):
    """Load trained policy from checkpoint."""
    from rsl_rl.runners import OnPolicyRunner

    ppo_runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    policy.eval()
    return policy


def _load_adaptation_module(checkpoint_path: str, device: str) -> AdaptationModule:
    """Load trained adaptation module."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["cfg"]
    cfg = AdaptationModuleCfg(**cfg_dict)
    model = AdaptationModule(cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded adaptation module:")
    print(f"       Input dim:  {cfg.in_dim}")
    print(f"       Latent dim: {cfg.latent_dim}")
    print(f"       Hidden:     {cfg.hidden_dims}")
    return model, cfg


def run_evaluation(args) -> EvalMetrics:
    """Run evaluation with adaptation module."""
    # Import Isaac Lab modules after SimulationApp starts
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import gymnasium as gym
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from agile.rl_env.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

    # Ensure environment registration
    import unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim  # noqa: F401

    # Load env + agent cfg
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "eval"):
        env_cfg.eval()

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device

    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # Load policy
    policy_checkpoint = Path(args.policy_checkpoint).expanduser()
    policy = _load_policy(str(policy_checkpoint), env, agent_cfg)
    print(f"[INFO] Loaded policy from: {policy_checkpoint}")

    # Load adaptation module
    adapt_checkpoint = Path(args.adaptation_checkpoint).expanduser()
    adapt_module, adapt_cfg = _load_adaptation_module(str(adapt_checkpoint), args.device)
    
    # Parse dimensions from adaptation config
    # hist_dim = hist_entry_dim * history_len
    # We need to infer history_len from the dataset metadata or from args
    latent_dim = adapt_cfg.latent_dim
    
    # Try to load metadata to get history parameters
    adapt_dir = adapt_checkpoint.parent.parent  # Go up from adaptation_model/
    meta_path = adapt_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        history_len = meta.get("history_len", args.history_len)
        hist_entry_dim = meta.get("hist_entry_dim", None)
        proprio_dim = meta.get("proprio_dim", None)
        z_t_stacked_dim = meta.get("z_t_stacked_dim", latent_dim * 5)
        print(f"[INFO] Loaded metadata from: {meta_path}")
        print(f"       History length: {history_len}")
        print(f"       Proprio dim:    {proprio_dim}")
        print(f"       Hist entry dim: {hist_entry_dim}")
    else:
        # Fallback: infer from observation dimensions
        history_len = args.history_len
        z_t_stacked_dim = latent_dim * 5  # 8 * 5 = 40
        proprio_dim = None
        hist_entry_dim = None
        print(f"[WARN] No metadata found, using defaults")
        print(f"       History length: {history_len}")

    # Reset env to get observation dimensions
    obs = _maybe_tuple_obs(env.reset())
    full_obs_dim = obs.shape[-1]
    act_dim = env.unwrapped.action_space.shape[0]
    num_envs = env.unwrapped.num_envs
    
    # Calculate proprio_dim if not from metadata
    if proprio_dim is None:
        proprio_dim = full_obs_dim - z_t_stacked_dim
    if hist_entry_dim is None:
        hist_entry_dim = proprio_dim + act_dim
    
    print(f"[INFO] Environment info:")
    print(f"       Num envs:      {num_envs}")
    print(f"       Full obs dim:  {full_obs_dim}")
    print(f"       Proprio dim:   {proprio_dim}")
    print(f"       Action dim:    {act_dim}")
    print(f"       z_t stacked:   {z_t_stacked_dim}")

    # Initialize history buffer
    history = torch.zeros(
        (num_envs, history_len, hist_entry_dim),
        device=env.unwrapped.device,
        dtype=torch.float32,
    )

    # Tracking for metrics
    metrics = EvalMetrics()
    episode_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
    episode_lengths = torch.zeros(num_envs, device=env.unwrapped.device, dtype=torch.long)
    
    print(f"\n[INFO] Starting evaluation for {args.num_steps} steps...")
    print(f"       Mode: {'Adaptation Module (student)' if not args.use_encoder else 'Encoder (teacher)'}")
    
    # Optional: load encoder for comparison
    encoder = None
    normalizer = None
    if args.use_encoder:
        from rma_modules import EnvFactorEncoder, EnvFactorEncoderCfg
        
        class EnvFactorNormalizer:
            def __init__(self, device):
                self.mins = torch.tensor([0.0] + [0.9]*12, device=device, dtype=torch.float32)
                self.maxs = torch.tensor([50.0] + [1.1]*12, device=device, dtype=torch.float32)
                self.ranges = self.maxs - self.mins
            def normalize(self, e_t):
                return (e_t - self.mins.to(e_t.device)) / (self.ranges.to(e_t.device) + 1e-8)
        
        # Try to find encoder checkpoint
        encoder_path = policy_checkpoint.parent / "checkpoints" / "encoder" / "encoder_latest.pt"
        if encoder_path.exists():
            encoder_cfg = EnvFactorEncoderCfg(in_dim=13, latent_dim=latent_dim)
            encoder = EnvFactorEncoder(cfg=encoder_cfg).to(args.device)
            encoder_ckpt = torch.load(encoder_path, map_location=args.device)
            encoder.load_state_dict(encoder_ckpt["model_state_dict"])
            encoder.eval()
            normalizer = EnvFactorNormalizer(args.device)
            print(f"[INFO] Loaded encoder from: {encoder_path}")

    with torch.no_grad():
        for step in range(args.num_steps):
            # Compute z_t based on mode
            if args.use_encoder and encoder is not None:
                # Teacher mode: use encoder with privileged e_t
                unwrapped = _get_unwrapped(env)
                e_t = getattr(unwrapped, "rma_env_factors_buf", None)
                if e_t is not None:
                    e_t = e_t[:, :13].to(args.device)
                    z_t = encoder(normalizer.normalize(e_t))
                else:
                    z_t = torch.zeros((num_envs, latent_dim), device=env.unwrapped.device)
            else:
                # Student mode: use adaptation module
                if step >= history_len - 1:
                    hist_flat = history.reshape(num_envs, -1).to(args.device)
                    z_t = adapt_module(hist_flat)
                else:
                    # Not enough history yet, use zeros
                    z_t = torch.zeros((num_envs, latent_dim), device=env.unwrapped.device)
            
            # Write z_t to environment for policy to use
            _write_extrinsics(env, z_t)

            # Get action from policy
            action = policy(obs)
            
            # Step environment
            next_obs, reward, done, info = _step_env(env, action)
            next_obs = _maybe_tuple_obs(next_obs)
            
            # Track metrics
            episode_rewards += reward.squeeze() if reward.ndim > 1 else reward
            episode_lengths += 1
            metrics.total_steps += num_envs
            
            # Handle episode ends
            if done is not None:
                done_mask = torch.as_tensor(done, device=history.device, dtype=torch.bool)
                if done_mask.any():
                    # Record completed episodes
                    for i in range(num_envs):
                        if done_mask[i]:
                            metrics.episode_rewards.append(episode_rewards[i].item())
                            metrics.episode_lengths.append(episode_lengths[i].item())
                            metrics.total_episodes += 1
                    
                    # Reset tracking for done envs
                    episode_rewards[done_mask] = 0.0
                    episode_lengths[done_mask] = 0
                    
                    # Clear history for done envs
                    history[done_mask] = 0.0

            # Update history buffer with proprioceptive observation (exclude z_t)
            proprio_obs = obs[:, :-z_t_stacked_dim]
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = torch.cat([proprio_obs, action], dim=-1)

            obs = next_obs
            
            # Progress logging
            if step > 0 and step % 500 == 0:
                print(f"[Step {step:5d}] Episodes: {metrics.total_episodes}, "
                      f"Mean Reward: {metrics.mean_reward:.2f}")

    env.close()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy RMA with adaptation module")
    parser.add_argument("--task", type=str, default="Unitree-H12-Walk-RMA-v0")
    parser.add_argument("--policy_checkpoint", type=str, required=True,
                        help="Path to trained policy checkpoint (.pt)")
    parser.add_argument("--adaptation_checkpoint", type=str, required=True,
                        help="Path to trained adaptation module checkpoint (.pt)")
    parser.add_argument("--history_len", type=int, default=20,
                        help="History length (should match training)")
    parser.add_argument("--num_steps", type=int, default=2000,
                        help="Number of evaluation steps")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel environments")
    parser.add_argument("--use_encoder", action="store_true",
                        help="Use encoder (teacher) instead of adaptation (student) for comparison")
    parser.add_argument("--seed", type=int, default=42)
    
    # Isaac Sim app args
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Launch simulation app
    simulation_app = None
    existing_kit_app = False
    try:
        import omni.kit_app
        existing_kit_app = omni.kit_app.get_app() is not None
    except Exception:
        existing_kit_app = False

    use_existing_kit_app = (
        args.use_existing_kit_app
        or existing_kit_app
        or os.environ.get("OMNI_APP_NAME")
        or os.environ.get("ISAACSIM_APP")
        or os.environ.get("OMNI_KIT_APP")
    )
    
    if not use_existing_kit_app:
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

    # Run evaluation
    print("=" * 70)
    print("RMA Deployment Evaluation")
    print("=" * 70)
    
    metrics = run_evaluation(args)
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(metrics.summary())
    print("=" * 70)

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
