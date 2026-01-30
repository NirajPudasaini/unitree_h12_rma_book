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

"""Standalone leg strength scaling visualizer for RMA training.

Demonstrates how reducing/increasing joint effort limits affects robot gait.
Useful for validating that the encoder learns to adapt policy under strength constraints.

Similar to visualize_rma_force.py but for e_t[1:13] (leg strength).
"""

# flake8: noqa

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Scale leg joint effort limits via RMA e_t[1:13].")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments.")
parser.add_argument("--steps", type=int, default=300, help="Steps to run.")
parser.add_argument(
    "--strength_scale",
    type=float,
    default=1.0,
    help="Multiplicative scale on leg joint effort limits (0.9=weak, 1.0=nominal, 1.1=strong).",
)
parser.add_argument(
    "--sweep",
    action="store_true",
    help="Sweep three scales: 0.9x (-10%), 1.0x (nominal), 1.1x (+10%) in separate environments.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import unitree_h12_sim2sim  # noqa: F401

from rma_modules.env_factor_spec import DEFAULT_ET_SPEC, LEG_JOINT_NAMES
from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim.mdp import rma as rma_mdp


# parse configuration
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

# Determine strength scales to test
if args_cli.sweep:
    strength_scales = [0.9, 1.0, 1.1]  # -10%, nominal, +10%
    if args_cli.num_envs < 3:
        print(f"[RMA] WARNING: --sweep requires at least 3 envs, but got {args_cli.num_envs}. Clamping to 3.", flush=True)
        env_cfg.num_envs = 3
else:
    strength_scales = [args_cli.strength_scale] * args_cli.num_envs

# disable automatic RMA sampling/printing events for this visualization
if hasattr(env_cfg, "events") and env_cfg.events is not None:
    for name in ("rma_env_factors", "rma_debug_print", "rma_debug_print_once", "rma_verify_apply_once"):
        if hasattr(env_cfg.events, name):
            delattr(env_cfg.events, name)

# create isaac environment
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
env_unwrapped = env.unwrapped

# Identify the robot and leg joints
robot = env_unwrapped.scene["robot"]
print(f"[RMA] Robot asset: {robot}", flush=True)
print(f"[RMA] Leg joint names: {LEG_JOINT_NAMES}", flush=True)

# Create a minimal asset_cfg that points to the robot
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
asset_cfg = SceneEntityCfg(name="robot")

# prepare actions
action_dim = int(getattr(env_unwrapped.action_manager, "total_action_dim", env.action_space.shape[0]))
zero_actions = torch.zeros((env_unwrapped.num_envs, action_dim), device=env_unwrapped.device)

# reset env
env.reset()

# prepare e_t buffer and env_ids
env_ids = torch.arange(env_unwrapped.num_envs, device=env_unwrapped.device)
et = rma_mdp._ensure_buffer(env_unwrapped, "rma_env_factors_buf", DEFAULT_ET_SPEC.dim)

# Build per-env strength scales (matching environments to scales)
strength_scale_per_env = torch.zeros((env_unwrapped.num_envs, DEFAULT_ET_SPEC.leg_strength_dim), device=env_unwrapped.device)
for i in range(env_unwrapped.num_envs):
    scale = strength_scales[i] if i < len(strength_scales) else strength_scales[-1]
    strength_scale_per_env[i, :] = scale

if args_cli.sweep:
    print(f"[RMA] SWEEP MODE: Testing {len(strength_scales)} strength scales:", flush=True)
    for i, scale in enumerate(strength_scales[:env_unwrapped.num_envs]):
        pct = (scale - 1.0) * 100
        label = f"{'WEAK (-10%)' if scale < 1.0 else 'NOMINAL' if scale == 1.0 else 'STRONG (+10%)'}"
        print(f"[RMA]   Env {i}: {scale:.2f}x ({pct:+.0f}%) {label}", flush=True)
else:
    print(f"[RMA] Setting all {DEFAULT_ET_SPEC.leg_strength_dim} leg joints to strength scale: {args_cli.strength_scale:.2f}x", flush=True)

# Apply strength scaling before first step
print(f"[RMA] Applying leg strength scaling...", flush=True)
try:
    rma_mdp._try_set_leg_effort_limits(env_unwrapped, asset_cfg, env_ids, strength_scale_per_env, LEG_JOINT_NAMES)
    print("[RMA] _try_set_leg_effort_limits SUCCESS", flush=True)
except Exception as e:
    print(f"[RMA] _try_set_leg_effort_limits FAILED: {e}", flush=True)

# Store strength in e_t[1:13]
et[env_ids, DEFAULT_ET_SPEC.leg_strength_slice] = strength_scale_per_env

# Readback baseline limits for comparison
baseline_limits = None
print(f"[RMA] Reading baseline effort limits...", flush=True)
try:
    baseline_limits, joint_ids = rma_mdp._read_leg_effort_limits(env_unwrapped, asset_cfg, env_ids, LEG_JOINT_NAMES)
    if baseline_limits is not None:
        print(f"[RMA] Baseline effort limits (mean per env):", flush=True)
        for i in range(min(3, env_unwrapped.num_envs)):
            print(f"[RMA]   Env {i}: {baseline_limits[i].mean().item():.3f} Nm", flush=True)
except Exception as e:
    print(f"[RMA] readback baseline limits failed: {e}", flush=True)

print(f"[RMA] Starting simulation with {args_cli.steps} steps (zero actions to see pure strength effect)...", flush=True)

# run simulation
for step_idx in range(args_cli.steps):
    if not simulation_app.is_running():
        break

    # Step the environment (keep zero actions so we see pure effect of strength changes)
    step_out = env.step(zero_actions)

    if step_idx == 0 or step_idx == 1:
        # Readback actual applied limits to verify
        try:
            limits, joint_ids = rma_mdp._read_leg_effort_limits(env_unwrapped, asset_cfg, env_ids, LEG_JOINT_NAMES)
            if limits is not None:
                print(f"[RMA] step {step_idx} actual effort limits (mean per env):", flush=True)
                for i in range(min(3, env_unwrapped.num_envs)):
                    actual_mean = limits[i].mean().item()
                    scale_i = strength_scales[i] if i < len(strength_scales) else strength_scales[-1]
                    expected_mean = (baseline_limits[i].mean().item() * scale_i) if baseline_limits is not None else None
                    pct = (scale_i - 1.0) * 100
                    print(f"[RMA]   Env {i} ({pct:+.0f}%): actual={actual_mean:.3f} Nm", flush=True)
                    if expected_mean is not None:
                        delta = actual_mean - expected_mean
                        print(f"[RMA]      expected (baseline * {scale_i:.2f}x={expected_mean:.3f}), delta={delta:+.4f} Nm", flush=True)
        except Exception as e:
            print(f"[RMA] readback failed: {e}", flush=True)

print("[RMA] Simulation complete.", flush=True)
if args_cli.sweep:
    print("[RMA]", flush=True)
    print("[RMA] SWEEP RESULTS: Compare gait differences across ±10% torque range", flush=True)
    print("[RMA]   Env 0 (0.9x, -10%): WEAK   → Robot should crouch/struggle", flush=True)
    print("[RMA]   Env 1 (1.0x,   0%): NOMINAL → Baseline behavior", flush=True)
    print("[RMA]   Env 2 (1.1x, +10%): STRONG → Robot should stride longer/taller", flush=True)
    print("[RMA]", flush=True)
    print("[RMA] Encoder learns: e_t[1:13] strength codes → z_t latent → policy adaptation", flush=True)
else:
    print(f"[RMA] Single scale test: strength_scale={args_cli.strength_scale:.2f}x", flush=True)
    if args_cli.strength_scale < 1.0:
        print("[RMA]   Expected: Robot may crouch/struggle due to limited torque", flush=True)
    elif args_cli.strength_scale > 1.0:
        print("[RMA]   Expected: Robot may stride longer/faster with extra torque headroom", flush=True)
    else:
        print("[RMA]   Expected: Normal nominal behavior", flush=True)

# close the simulator
env.close()
simulation_app.close()
