#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase-2 adaptation: collect history->z_t dataset and train adaptation module."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add local packages to PYTHONPATH
workspace_root = Path(__file__).resolve().parents[1]
isaaclab_root = Path("/home/niraj/isaac_projects/IsaacLab")
sys.path.insert(0, str(isaaclab_root / "source"))
sys.path.insert(0, str(workspace_root / "isaaclab" / "source"))
sys.path.insert(0, str(workspace_root / "rma"))
sys.path.insert(0, str(workspace_root / "isaaclab" / "scripts" / "rsl_rl"))

from rma_modules import (
    AdaptationModule,
    AdaptationModuleCfg,
    EnvFactorEncoder,
    EnvFactorEncoderCfg,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class EnvFactorNormalizer:
    """Normalizes environment factors to [0, 1] (force + leg_strength only, 13 dims)."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.mins = torch.tensor(
            [
                0.0,  # payload force
                *([0.9] * 12),  # leg strengths
            ],
            device=device,
            dtype=torch.float32,
        )
        self.maxs = torch.tensor(
            [
                50.0,  # payload force
                *([1.1] * 12),  # leg strengths
            ],
            device=device,
            dtype=torch.float32,
        )
        self.ranges = self.maxs - self.mins

    def normalize(self, e_t: torch.Tensor) -> torch.Tensor:
        mins = self.mins.to(e_t.device)
        ranges = self.ranges.to(e_t.device)
        return (e_t - mins) / (ranges + 1e-8)


def _get_unwrapped(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _maybe_tuple_obs(obs):
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def _step_env(env, action):
    step_out = env.step(action)
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = terminated | truncated
        return obs, reward, done, info
    obs, reward, done, info = step_out
    return obs, reward, done, info


def _write_extrinsics(env, z_t: torch.Tensor) -> None:
    """Write z_t into env.rma_extrinsics_buf (creates if missing)."""
    unwrapped = _get_unwrapped(env)
    z_buf = getattr(unwrapped, "rma_extrinsics_buf", None)
    if z_buf is None or z_buf.shape != (unwrapped.num_envs, z_t.shape[-1]):
        z_buf = torch.zeros((unwrapped.num_envs, z_t.shape[-1]), device=unwrapped.device, dtype=torch.float)
        setattr(unwrapped, "rma_extrinsics_buf", z_buf)
    z_buf.copy_(z_t.to(unwrapped.device))


def _load_policy(checkpoint_path: str, env, agent_cfg):
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
    return policy, ppo_runner


def _resolve_encoder_checkpoint(policy_checkpoint: Path, encoder_checkpoint: str | None) -> Path:
    if encoder_checkpoint:
        return Path(encoder_checkpoint).expanduser()
    # Infer from policy checkpoint directory
    run_dir = policy_checkpoint.parent
    candidate = run_dir / "checkpoints" / "encoder" / "encoder_latest.pt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Encoder checkpoint not found. Provide --encoder_checkpoint explicitly."
    )


def _save_metadata(output_dir: Path, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def collect_dataset(args) -> Path:
    """Collect (history -> z_t) dataset for adaptation."""
    # NOTE: Do not import any Isaac Sim/Omni modules before SimulationApp starts.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import gymnasium as gym
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from agile.rl_env.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

    # Ensure environment registration is executed after SimulationApp starts
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

    policy_checkpoint = Path(args.policy_checkpoint).expanduser()
    policy, _ppo_runner = _load_policy(str(policy_checkpoint), env, agent_cfg)

    encoder_ckpt_path = _resolve_encoder_checkpoint(policy_checkpoint, args.encoder_checkpoint)
    encoder_cfg = EnvFactorEncoderCfg(in_dim=13, latent_dim=8, hidden_dims=(256, 128))
    encoder = EnvFactorEncoder(cfg=encoder_cfg).to(args.device)
    encoder_ckpt = torch.load(encoder_ckpt_path, map_location=args.device)
    encoder.load_state_dict(encoder_ckpt["model_state_dict"])
    encoder.eval()

    normalizer = EnvFactorNormalizer(device=args.device)

    # Reset env
    obs = _maybe_tuple_obs(env.reset())
    obs_dim = obs.shape[-1]
    act_dim = env.unwrapped.action_space.shape[0]
    num_envs = env.unwrapped.num_envs
    hist_dim = (obs_dim + act_dim) * args.history_len

    history = torch.zeros(
        (num_envs, args.history_len, obs_dim + act_dim),
        device=env.unwrapped.device,
        dtype=torch.float32,
    )
    # Optional adaptation rollout module (random init)
    adapt_module = None
    adapt_optimizer = None
    if args.rollout_extrinsics == "adaptation":
        adapt_cfg = AdaptationModuleCfg(
            in_dim=hist_dim,
            latent_dim=args.latent_dim,
            hidden_dims=(256, 128),
            activation="elu",
        )
        adapt_module = AdaptationModule(cfg=adapt_cfg).to(args.device)
        adapt_module.train()
        if args.online_update:
            adapt_optimizer = torch.optim.Adam(adapt_module.parameters(), lr=args.online_lr)

    output_dir = Path(args.dataset_dir).expanduser() if args.dataset_dir else (
        workspace_root / "rma" / "outputs" / "adaptation_dataset" / _timestamp()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "task": args.task,
        "history_len": args.history_len,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "hist_dim": hist_dim,
        "num_envs": num_envs,
        "num_steps": args.num_steps,
        "policy_checkpoint": str(policy_checkpoint),
        "encoder_checkpoint": str(encoder_ckpt_path),
        "rollout_extrinsics": args.rollout_extrinsics,
        "online_update": bool(args.online_update),
        "created_at": datetime.now().isoformat(),
    }
    _save_metadata(output_dir, meta)

    chunk_idx = 0
    chunk_histories = []
    chunk_targets = []

    with torch.no_grad():
        for step in range(args.num_steps):
            # Compute teacher z_t from privileged e_t
            unwrapped = _get_unwrapped(env)
            e_t = getattr(unwrapped, "rma_env_factors_buf", None)
            if e_t is not None:
                e_t = e_t[:, :13].to(args.device)
                z_t_teacher = encoder(normalizer.normalize(e_t))
            else:
                z_t_teacher = None

            # Write extrinsics for rollout
            if args.rollout_extrinsics == "encoder" and z_t_teacher is not None:
                _write_extrinsics(env, z_t_teacher)
            elif args.rollout_extrinsics == "adaptation" and adapt_module is not None and step >= args.history_len - 1:
                hist_flat = history.reshape(num_envs, -1).to(args.device)
                z_hat = adapt_module(hist_flat)
                _write_extrinsics(env, z_hat)
            elif args.rollout_extrinsics == "zero":
                _write_extrinsics(env, torch.zeros((num_envs, args.latent_dim), device=env.unwrapped.device))

            action = policy(obs)
            next_obs, _reward, done, _info = _step_env(env, action)
            next_obs = _maybe_tuple_obs(next_obs)

            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1, :] = torch.cat([obs, action], dim=-1)

            # Clear history for reset environments
            if done is not None:
                done_mask = torch.as_tensor(done, device=history.device, dtype=torch.bool)
                if done_mask.any():
                    history[done_mask] = 0.0

            if step >= args.history_len - 1:
                if z_t_teacher is None:
                    continue

                flat_hist = history.reshape(num_envs, -1).detach().cpu()
                chunk_histories.append(flat_hist)
                chunk_targets.append(z_t_teacher.detach().cpu())

                # Optional online update of adaptation module
                if args.rollout_extrinsics == "adaptation" and args.online_update and adapt_module is not None:
                    adapt_optimizer.zero_grad(set_to_none=True)
                    pred = adapt_module(history.reshape(num_envs, -1).to(args.device))
                    loss = torch.nn.functional.mse_loss(pred, z_t_teacher.detach())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapt_module.parameters(), max_norm=1.0)
                    adapt_optimizer.step()

                if len(chunk_histories) >= args.save_every:
                    _write_chunk(output_dir, chunk_idx, chunk_histories, chunk_targets)
                    chunk_idx += 1
                    chunk_histories, chunk_targets = [], []

            obs = next_obs

    if chunk_histories:
        _write_chunk(output_dir, chunk_idx, chunk_histories, chunk_targets)

    env.close()

    return output_dir


def _write_chunk(output_dir: Path, chunk_idx: int, histories, targets) -> None:
    chunk_hist = torch.cat(histories, dim=0)
    chunk_tgt = torch.cat(targets, dim=0)
    out_path = output_dir / f"chunk_{chunk_idx:04d}.pt"
    torch.save({"history": chunk_hist, "z_t": chunk_tgt}, out_path)


def _load_dataset_chunks(dataset_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = sorted(dataset_dir.glob("chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No chunk_*.pt files found in {dataset_dir}")
    histories = []
    targets = []
    for path in chunks:
        data = torch.load(path, map_location="cpu")
        histories.append(data["history"])
        targets.append(data["z_t"])
    return torch.cat(histories, dim=0), torch.cat(targets, dim=0)


def train_adaptation(args, dataset_dir: Path) -> Path:
    """Train adaptation module on collected dataset."""
    histories, targets = _load_dataset_chunks(dataset_dir)
    input_dim = histories.shape[-1]
    latent_dim = targets.shape[-1]

    cfg = AdaptationModuleCfg(
        in_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=(256, 128),
        activation="elu",
    )
    model = AdaptationModule(cfg=cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(histories, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch_hist, batch_tgt in loader:
            batch_hist = batch_hist.to(args.device)
            batch_tgt = batch_tgt.to(args.device)
            pred = model(batch_hist)
            loss = loss_fn(pred, batch_tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(loader))
        print(f"[Epoch {epoch:03d}] loss={avg_loss:.6f}")

    output_dir = dataset_dir / "adaptation_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "adaptation_latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": asdict(cfg),
            "dataset_dir": str(dataset_dir),
        },
        out_path,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-2 adaptation dataset + training")
    parser.add_argument("--mode", choices=("collect", "train", "both"), default="both")
    parser.add_argument("--task", type=str, default="Unitree-H12-Walk-RMA-v0")
    parser.add_argument("--policy_checkpoint", type=str, default=None, help="Policy checkpoint (.pt)")
    parser.add_argument("--encoder_checkpoint", type=str, default=None, help="Encoder checkpoint (.pt)")
    parser.add_argument("--history_len", type=int, default=50, help="History length H")
    parser.add_argument("--num_steps", type=int, default=5000, help="Steps to collect for dataset")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Output dataset directory")
    parser.add_argument("--save_every", type=int, default=200, help="Chunks per save (in steps)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent z dimension")
    parser.add_argument(
        "--rollout_extrinsics",
        type=str,
        choices=("encoder", "adaptation", "zero"),
        default="encoder",
        help="Which extrinsics to feed into policy during collection",
    )
    parser.add_argument(
        "--online_update",
        action="store_true",
        help="If set, update adaptation module online during collection",
    )
    parser.add_argument("--online_lr", type=float, default=1e-3, help="LR for online updates")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Isaac Sim app args
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # Launch omniverse app (or reuse existing Kit app), same pattern as Phase-1.
    simulation_app = None
    existing_kit_app = False
    try:
        import omni.kit_app  # type: ignore

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

    dataset_dir = None
    if args.mode in ("collect", "both"):
        if args.policy_checkpoint is None:
            raise ValueError("--policy_checkpoint is required for collect mode.")
        dataset_dir = collect_dataset(args)
        print(f"[INFO] Saved dataset to: {dataset_dir}")
    else:
        dataset_dir = Path(args.dataset_dir).expanduser() if args.dataset_dir else None

    if args.mode in ("train", "both"):
        if dataset_dir is None:
            raise ValueError("--dataset_dir is required for train mode.")
        model_path = train_adaptation(args, dataset_dir)
        print(f"[INFO] Saved adaptation model to: {model_path}")

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
