# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to train RMA policy with encoder and decoder."""

"""Launch Isaac Sim Simulator first."""

# flake8: noqa

import argparse
import sys
from pathlib import Path

# Add isaaclab scripts directory to path for cli_args import
sys.path.insert(0, str(Path(__file__).parent.parent / "isaaclab" / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RMA RL agent with encoder/decoder.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-H12-Walk-RMA-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--encoder_loss_weight", type=float, default=0.01, help="Weight for encoder reconstruction loss.")
parser.add_argument("--decoder_loss_weight", type=float, default=0.1, help="Weight for decoder reconstruction loss.")
parser.add_argument("--save_encoder_every", type=int, default=10000, help="Save encoder every N iterations.")
parser.add_argument("--save_decoder_every", type=int, default=10000, help="Save decoder every N iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse known and unknown arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import agile.isaaclab_extras.monkey_patches

# PLACEHOLDER: Extension template (do not remove this comment)
import unitree_h12_sim2sim  # noqa: F401

from rma_modules import (  # isort: skip
    EnvFactorEncoder,
    EnvFactorEncoderCfg,
    EnvFactorDecoder,
    EnvFactorDecoderCfg,
)

from agile.rl_env.rsl_rl import (  # isort: skip
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class EnvFactorNormalizer:
    """Normalizes environment factors to [0, 1] range for better neural network training."""
    
    def __init__(self, device: str = "cpu"):
        """Initialize normalizer with predefined ranges for 17D environment factors.
        
        Args:
            device: Torch device.
        """
        self.device = device
        
        # Define min/max ranges for each of the 17 environment factors
        # Factor order: [payload_force, leg_strength_0-11, friction, terrain_amp, terrain_lengthscale, terrain_noise]
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
        """Normalize environment factors to [0, 1].
        
        Args:
            e_t: Environment factors tensor of shape (..., 17)
            
        Returns:
            Normalized tensor of same shape, values in [0, 1]
        """
        return (e_t - self.mins) / (self.ranges + 1e-8)
    
    def denormalize(self, e_t_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0, 1] back to original ranges.
        
        Args:
            e_t_normalized: Normalized environment factors of shape (..., 17)
            
        Returns:
            Denormalized tensor with original ranges
        """
        ranges = self.ranges.to(e_t_normalized.device)
        mins = self.mins.to(e_t_normalized.device)
        return e_t_normalized * ranges + mins


class RMATrainerWithEncoding:
    """Trainer for RMA policy with encoder and decoder networks."""

    def __init__(
        self,
        runner: OnPolicyRunner,
        encoder: EnvFactorEncoder,
        decoder: EnvFactorDecoder,
        log_dir: str,
        device: str,
        encoder_loss_weight: float = 0.01,
        decoder_loss_weight: float = 0.1,
        save_encoder_every: int = 100,
        save_decoder_every: int = 100,
    ):
        """Initialize the trainer.

        Args:
            runner: RSL-RL OnPolicyRunner for policy training.
            encoder: EnvFactorEncoder network.
            decoder: EnvFactorDecoder network.
            log_dir: Directory for logging.
            device: Torch device.
            encoder_loss_weight: Weight for encoder auxiliary loss.
            decoder_loss_weight: Weight for decoder reconstruction loss.
            save_encoder_every: Save encoder checkpoint every N iterations.
            save_decoder_every: Save decoder checkpoint every N iterations.
        """
        self.runner = runner
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.log_dir = log_dir
        self.device = device
        
        # Initialize normalizer for environment factors
        self.normalizer = EnvFactorNormalizer(device=device)

        self.encoder_loss_weight = encoder_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.save_encoder_every = save_encoder_every
        self.save_decoder_every = save_decoder_every

        # Optimizers for encoder and decoder
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3)

        # Create checkpoint directories
        self.encoder_dir = Path(log_dir) / "checkpoints" / "encoder"
        self.decoder_dir = Path(log_dir) / "checkpoints" / "decoder"
        self.encoder_dir.mkdir(parents=True, exist_ok=True)
        self.decoder_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.encoder_losses = []
        self.decoder_losses = []
        
        # Buffer for storing e_t samples from environment
        self.e_t_buffer = []
        
        # Original learn method reference
        self._original_learn = None

    def collect_e_t_from_env(self, env) -> torch.Tensor:
        """Extract RMA environment factors directly from environment internal state.

        The environment sampled e_t at the last reset and stores it in rma_env_factors.

        Args:
            env: The environment instance.

        Returns:
            e_t tensor of shape (num_envs, 17), or None if unable to extract.
        """
        # Try to access RMA factors through the environment hierarchy
        # RslRlVecEnvWrapper -> env -> H12LocomotionFullBodyRmaEnv
        env_to_check = env
        
        # Unwrap RslRlVecEnvWrapper
        if hasattr(env_to_check, "env"):
            env_to_check = env_to_check.env
        
        # Unwrap gym wrappers
        while hasattr(env_to_check, "env"):
            env_to_check = env_to_check.env
        
        # Now env_to_check should be the actual IsaacLab environment
        if hasattr(env_to_check, "scene") and hasattr(env_to_check.scene, "rma_env_factors"):
            e_t = env_to_check.scene.rma_env_factors
            if isinstance(e_t, torch.Tensor):
                return e_t.to(self.device).clone()
        
        # Alternative: check if env has state dict or module with rma_env_factors
        for attr_name in ["state", "_state", "unwrapped"]:
            if hasattr(env_to_check, attr_name):
                attr = getattr(env_to_check, attr_name)
                if hasattr(attr, "rma_env_factors"):
                    e_t = attr.rma_env_factors
                    if isinstance(e_t, torch.Tensor):
                        return e_t.to(self.device).clone()
        
        # Last resort: search all attributes for rma_env_factors
        try:
            for attr_name in dir(env_to_check):
                if "rma_env_factors" in attr_name.lower() or "rma_factors" in attr_name.lower():
                    attr = getattr(env_to_check, attr_name, None)
                    if isinstance(attr, torch.Tensor) and attr.shape[-1] == 17:
                        return attr.to(self.device).clone()
        except:
            pass
        
        return None
    
    def store_e_t_samples(self, env):
        """Store environment factor samples during policy training.
        
        Args:
            env: The environment instance.
        """
        e_t = self.collect_e_t_from_env(env)
        if e_t is not None and e_t.numel() > 0:
            self.e_t_buffer.append(e_t.cpu())

    def train_step(self, e_t_batch: torch.Tensor, iteration: int):
        """Perform one training step for encoder and decoder.

        Args:
            e_t_batch: Ground-truth RMA factors of shape (batch_size, 17).
            iteration: Current training iteration.
        """
        # Ensure e_t is on device
        e_t = e_t_batch.to(self.device)
        
        if e_t is None or e_t.shape[0] == 0:
            return  # Skip if unable to extract

        # Normalize environment factors to [0, 1]
        e_t_normalized = self.normalizer.normalize(e_t)

        # Forward pass through encoder: e_t_normalized -> z_t
        z_t = self.encoder(e_t_normalized)

        # Forward pass through decoder: z_t -> ê_t_normalized (no scaling needed, already normalized)
        e_t_reconstructed = self.decoder(z_t, apply_scaling=False)

        # Compute decoder loss (reconstruction) - on normalized values
        decoder_loss = self.decoder.compute_reconstruction_loss(z_t, e_t_normalized)
        self.decoder_losses.append(decoder_loss.item())

        # Backward pass for decoder
        self.decoder_optimizer.zero_grad()
        decoder_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        self.decoder_optimizer.step()

        # Optional: Compute encoder loss (could be info bottleneck or other auxiliary loss)
        # For now, we just track reconstruction quality
        encoder_loss = torch.nn.functional.mse_loss(z_t, torch.zeros_like(z_t))  # Placeholder
        self.encoder_losses.append(encoder_loss.item())

        # Log losses every 10 iterations
        if iteration % 10 == 0:
            avg_encoder_loss = sum(self.encoder_losses[-10:]) / min(10, len(self.encoder_losses))
            avg_decoder_loss = sum(self.decoder_losses[-10:]) / min(10, len(self.decoder_losses))
            print(
                f"[Encoder/Decoder Iteration {iteration}] "
                f"Encoder Loss: {avg_encoder_loss:.6f} | "
                f"Decoder Loss: {avg_decoder_loss:.6f}"
            )
        
        # Note: Checkpoints are saved in the main train() loop, not here

    def save_encoder(self, iteration: int):
        """Save encoder checkpoint.

        Args:
            iteration: Current training iteration.
        """
        try:
            checkpoint_path = self.encoder_dir / f"encoder_iter_{iteration:06d}.pt"
            torch.save(
                {
                    "model_state_dict": self.encoder.state_dict(),
                    "config": self.encoder.cfg.__dict__ if hasattr(self.encoder.cfg, "__dict__") else {},
                    "iteration": iteration,
                },
                checkpoint_path,
            )
            print(f"[INFO] Saved encoder checkpoint: {checkpoint_path}")

            # Also save as latest
            latest_path = self.encoder_dir / "encoder_latest.pt"
            torch.save(
                {
                    "model_state_dict": self.encoder.state_dict(),
                    "config": self.encoder.cfg.__dict__ if hasattr(self.encoder.cfg, "__dict__") else {},
                    "iteration": iteration,
                },
                latest_path,
            )
            print(f"[INFO] Saved encoder_latest.pt")
        except Exception as e:
            print(f"[ERROR] Failed to save encoder checkpoint: {e}")

    def save_decoder(self, iteration: int):
        """Save decoder checkpoint.

        Args:
            iteration: Current training iteration.
        """
        try:
            checkpoint_path = self.decoder_dir / f"decoder_iter_{iteration:06d}.pt"
            torch.save(
                {
                    "model_state_dict": self.decoder.state_dict(),
                    "config": self.decoder.cfg.__dict__ if hasattr(self.decoder.cfg, "__dict__") else {},
                    "iteration": iteration,
                },
                checkpoint_path,
            )
            print(f"[INFO] Saved decoder checkpoint: {checkpoint_path}")

            # Also save as latest
            latest_path = self.decoder_dir / "decoder_latest.pt"
            torch.save(
                {
                    "model_state_dict": self.decoder.state_dict(),
                    "config": self.decoder.cfg.__dict__ if hasattr(self.decoder.cfg, "__dict__") else {},
                    "iteration": iteration,
                },
                latest_path,
            )
            print(f"[INFO] Saved decoder_latest.pt")
        except Exception as e:
            print(f"[ERROR] Failed to save decoder checkpoint: {e}")
    
    def collect_samples_during_training(self):
        """Periodically collect e_t samples from the environment during policy training.
        
        This should be called periodically during the policy training loop.
        """
        env = self.runner.env
        e_t = self.collect_e_t_from_env(env)
        if e_t is not None and e_t.numel() > 0:
            self.e_t_buffer.append(e_t.cpu())
    
    def hook_environment_for_sample_collection(self):
        """Prepare for periodic sample collection during training.
        
        Note: Instead of hooking methods, we'll collect samples periodically 
        in the main loop to ensure reliable collection.
        """
        print("[INFO] Prepared for periodic e_t sample collection during training")

    def collect_samples_batch(self, num_batches: int = 10):
        """Collect multiple batches of samples.
        
        Args:
            num_batches: Number of collection attempts.
        """
        collected = 0
        for _ in range(num_batches):
            e_t = self.collect_e_t_from_env(self.runner.env)
            if e_t is not None and e_t.numel() > 0:
                self.e_t_buffer.append(e_t.cpu())
                collected += 1
        return collected

    def train(self, num_iterations: int):
        """Train encoder and decoder using collected e_t samples.

        Args:
            num_iterations: Target number of training iterations (will iterate through batches multiple times if needed).
        """
        if len(self.e_t_buffer) == 0:
            print("[WARNING] No e_t samples in buffer. Skipping encoder/decoder training.")
            print("[INFO] Make sure to call store_e_t_samples() during policy training.")
            # Still save models with random weights for structure validation
            self.save_encoder(0)
            self.save_decoder(0)
            return

        # Concatenate all buffered samples
        e_t_all = torch.cat(self.e_t_buffer, dim=0)  # Shape: (total_samples, 17)
        print(f"[INFO] Training encoder/decoder with {e_t_all.shape[0]} total environment factor samples")

        # Training setup
        batch_size = 256
        num_samples = e_t_all.shape[0]
        num_batches_per_epoch = max(1, num_samples // batch_size)
        num_epochs = max(1, (num_iterations + num_batches_per_epoch - 1) // num_batches_per_epoch)
        
        print(f"[INFO] Encoder/decoder training configuration:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Num samples: {num_samples}")
        print(f"  - Batches per epoch: {num_batches_per_epoch}")
        print(f"  - Target iterations: {num_iterations}")
        print(f"  - Calculated epochs: {num_epochs}")

        total_iterations = 0
        print(f"[INFO] Starting auxiliary encoder/decoder training for {num_epochs} epochs...")
        
        # Save initial checkpoint
        self.save_encoder(0)
        self.save_decoder(0)
        print(f"[INFO] Saved initial checkpoint at iteration 0")
        
        for epoch in range(1, num_epochs + 1):
            # Shuffle indices
            indices = torch.randperm(num_samples)
            e_t_shuffled = e_t_all[indices]

            # Iterate over batches in this epoch
            for batch_idx in range(num_batches_per_epoch):
                # Stop if we've reached target iterations
                if total_iterations >= num_iterations:
                    break
                    
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                e_t_batch = e_t_shuffled[start_idx:end_idx]

                # Train step
                self.train_step(e_t_batch, total_iterations)
                total_iterations += 1

                # Save checkpoint periodically
                if total_iterations % self.save_decoder_every == 0:
                    self.save_decoder(total_iterations)
                if total_iterations % self.save_encoder_every == 0:
                    self.save_encoder(total_iterations)

            # Early exit if we've reached iterations
            if total_iterations >= num_iterations:
                break
            
            print(f"[INFO] Epoch {epoch}/{num_epochs} completed ({total_iterations} total iterations)")

        print(f"[INFO] Encoder/decoder training completed!")
        print(f"[INFO] Total training iterations: {total_iterations}")
        
        # Final save
        self.save_encoder(total_iterations)
        self.save_decoder(total_iterations)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Train RMA policy with encoder and decoder."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # update frequency in case it is overridden
    if any("physics_freq" in arg or "controller_freq" in arg for arg in hydra_args):
        env_cfg.decimation = int(env_cfg.physics_freq / env_cfg.controller_freq)
        env_cfg.sim.dt = 1.0 / env_cfg.physics_freq
        env_cfg.sim.render_interval = env_cfg.decimation
        for attr_name in dir(env_cfg.scene):
            attr = getattr(env_cfg.scene, attr_name)
            if hasattr(attr, "update_period"):
                prev_update_period = attr.update_period
                if hasattr(attr, "force_threshold"):
                    attr.update_period = 1.0 / env_cfg.physics_freq
                else:
                    attr.update_period = 1.0 / env_cfg.controller_freq
                print(f"{attr_name} update period changed from {prev_update_period} to {attr.update_period}")

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # specify directory for logging runs
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_interval_steps = args_cli.video_interval_iter * agent_cfg.num_steps_per_env
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % video_interval_steps == 0,
            "video_length": 200,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # load the checkpoint if resuming
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, load_optimizer=agent_cfg.load_optimizer)

    # Create encoder and decoder
    encoder_cfg = EnvFactorEncoderCfg(in_dim=17, latent_dim=8, hidden_dims=(256, 128))
    encoder = EnvFactorEncoder(cfg=encoder_cfg)

    decoder_cfg = EnvFactorDecoderCfg()
    decoder = EnvFactorDecoder(cfg=decoder_cfg)

    # Create trainer
    trainer = RMATrainerWithEncoding(
        runner=runner,
        encoder=encoder,
        decoder=decoder,
        log_dir=log_dir,
        device=agent_cfg.device,
        encoder_loss_weight=args_cli.encoder_loss_weight,
        decoder_loss_weight=args_cli.decoder_loss_weight,
        save_encoder_every=args_cli.save_encoder_every,
        save_decoder_every=args_cli.save_decoder_every,
    )

    # Print configuration
    print("[INFO] RMA Training Configuration:")
    print(f"  - Encoder input dim: 17 (environment factors)")
    print(f"  - Encoder output dim: 8 (latent)")
    print(f"  - Decoder input dim: 8 (latent)")
    print(f"  - Decoder output dim: 17 (reconstructed factors)")
    print(f"  - Max iterations: {agent_cfg.max_iterations}")
    print(f"  - Encoder loss weight: {args_cli.encoder_loss_weight}")
    print(f"  - Decoder loss weight: {args_cli.decoder_loss_weight}")

    # Run training loop with concurrent encoder/decoder training
    print("\n[INFO] Starting policy training with concurrent encoder/decoder auxiliary training...")
    
    # Manual policy training loop with encoder/decoder training integrated
    print(f"[INFO] Running policy for {agent_cfg.max_iterations} iterations with concurrent encoder/decoder training...")
    print(f"[INFO] Encoder/decoder will train every 50 policy iterations using accumulated samples...")
    
    # Track iterations for encoder/decoder training
    encoder_decoder_train_interval = 50  # Train encoder/decoder every N policy iterations
    policy_iterations_completed = 0
    encoder_decoder_iterations_completed = 0
    
    # Get the internal runner to access learning function properly
    for policy_iter in range(agent_cfg.max_iterations):
        # Run one policy learning iteration
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
        policy_iterations_completed += 1
        
        # Collect e_t samples from environment after each policy iteration
        for _ in range(20):  # Collect multiple batches per iteration
            e_t = trainer.collect_e_t_from_env(runner.env)
            if e_t is not None and e_t.numel() > 0:
                trainer.e_t_buffer.append(e_t.cpu())
        
        # Train encoder/decoder periodically
        if (policy_iter + 1) % encoder_decoder_train_interval == 0 and len(trainer.e_t_buffer) > 0:
            # Train encoder/decoder for a few iterations on accumulated samples
            print(f"\n[INFO] Policy iteration {policy_iter + 1}/{agent_cfg.max_iterations} - Training encoder/decoder...")
            
            # Collect e_t samples into tensor
            e_t_all = torch.cat(trainer.e_t_buffer, dim=0)
            print(f"[INFO] Training on {e_t_all.shape[0]} samples")
            
            # Train for a few encoder/decoder iterations
            batch_size = 256
            num_samples = e_t_all.shape[0]
            
            # Save initial checkpoint (first time only)
            if encoder_decoder_iterations_completed == 0:
                trainer.save_encoder(0)
                trainer.save_decoder(0)
                print(f"[INFO] Saved initial checkpoint at iteration 0")
            
            for ed_step in range(min(50, max(1, num_samples // batch_size))):  # Train for up to 50 steps
                indices = torch.randperm(num_samples)
                e_t_shuffled = e_t_all[indices]
                
                for batch_idx in range(0, num_samples, batch_size):
                    end_idx = min(batch_idx + batch_size, num_samples)
                    e_t_batch = e_t_shuffled[batch_idx:end_idx]
                    
                    # Train step
                    trainer.train_step(e_t_batch, encoder_decoder_iterations_completed)
                    encoder_decoder_iterations_completed += 1
                    
                    # Save checkpoint periodically
                    if encoder_decoder_iterations_completed % trainer.save_encoder_every == 0:
                        trainer.save_encoder(encoder_decoder_iterations_completed)
                        print(f"[INFO] Saved encoder at iteration {encoder_decoder_iterations_completed}")
                    if encoder_decoder_iterations_completed % trainer.save_decoder_every == 0:
                        trainer.save_decoder(encoder_decoder_iterations_completed)
                        print(f"[INFO] Saved decoder at iteration {encoder_decoder_iterations_completed}")
            
            # Clear buffer for next batch
            trainer.e_t_buffer.clear()
            print(f"[INFO] Encoder/decoder training complete for this batch. Total ED iterations: {encoder_decoder_iterations_completed}")
    
    print(f"[INFO] Policy training completed.")
    
    # Final encoder/decoder training on any remaining samples
    if len(trainer.e_t_buffer) > 0:
        print("\n[INFO] Final encoder/decoder training on remaining samples...")
        e_t_all = torch.cat(trainer.e_t_buffer, dim=0)
        print(f"[INFO] Training on {e_t_all.shape[0]} remaining samples")
        
        batch_size = 256
        num_samples = e_t_all.shape[0]
        
        for ed_step in range(min(50, max(1, num_samples // batch_size))):
            indices = torch.randperm(num_samples)
            e_t_shuffled = e_t_all[indices]
            
            for batch_idx in range(0, num_samples, batch_size):
                end_idx = min(batch_idx + batch_size, num_samples)
                e_t_batch = e_t_shuffled[batch_idx:end_idx]
                
                trainer.train_step(e_t_batch, encoder_decoder_iterations_completed)
                encoder_decoder_iterations_completed += 1
                
                if encoder_decoder_iterations_completed % trainer.save_encoder_every == 0:
                    trainer.save_encoder(encoder_decoder_iterations_completed)
                if encoder_decoder_iterations_completed % trainer.save_decoder_every == 0:
                    trainer.save_decoder(encoder_decoder_iterations_completed)
        
        trainer.e_t_buffer.clear()
    
    # Final checkpoint save
    trainer.save_encoder(encoder_decoder_iterations_completed)
    trainer.save_decoder(encoder_decoder_iterations_completed)
    print(f"[INFO] Final encoder/decoder checkpoints saved at iteration {encoder_decoder_iterations_completed}")

    # close the simulator
    env.close()

    print(f"[INFO] Training complete! Logs saved to: {log_dir}")
    print(f"[INFO] Encoder checkpoints saved to: {trainer.encoder_dir}")
    print(f"[INFO] Decoder checkpoints saved to: {trainer.decoder_dir}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
