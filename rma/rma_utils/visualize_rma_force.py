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

"""Minimal external wrench visualizer for RMA training.

Follows the documented Isaac Lab external wrench workflow:
1. Find torso_link body_ids
2. Sample horizontal pushes
3. Call set_external_force_and_torque
4. Call write_data_to_sim() BEFORE env.step()
"""

# flake8: noqa

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Apply external wrench to torso_link via RMA e_t[0].")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=1000, help="Steps to run.")

parser.add_argument(
    "--push_range",
    type=float,
    nargs=2,
    default=(0.0, 50.0), # ~ around 5kgs
    help="Min/max constant downward force (N). ~1 N per kg.",
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

from rma_modules.env_factor_spec import DEFAULT_ET_SPEC
from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim.mdp import rma as rma_mdp


# parse configuration
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

# disable automatic RMA sampling/printing events for this visualization
if hasattr(env_cfg, "events") and env_cfg.events is not None:
    for name in ("rma_env_factors", "rma_debug_print", "rma_debug_print_once", "rma_verify_apply_once"):
        if hasattr(env_cfg.events, name):
            delattr(env_cfg.events, name)

# create isaac environment
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
env_unwrapped = env.unwrapped

# Identify the body to be pushed: torso_link
robot = env_unwrapped.scene["robot"]
body_ids, _ = robot.find_bodies("torso_link")
body_ids = torch.tensor(body_ids, device=env_unwrapped.device)

print(f"[RMA] Found torso_link body_ids: {body_ids}", flush=True)
print(f"[RMA] Robot asset: {robot}", flush=True)
print(f"[RMA] Has set_external_force_and_torque: {hasattr(robot, 'set_external_force_and_torque')}", flush=True)
print(f"[RMA] Has write_data_to_sim: {hasattr(robot, 'write_data_to_sim')}", flush=True)

# prepare actions
action_dim = int(getattr(env_unwrapped.action_manager, "total_action_dim", env.action_space.shape[0]))
zero_actions = torch.zeros((env_unwrapped.num_envs, action_dim), device=env_unwrapped.device)

# Disable the action manager so external force is not overpowered by controller
env_unwrapped.action_manager.set_debug_vis(False)
if hasattr(env_unwrapped, "_action_manager"):
    env_unwrapped._action_manager.is_enabled = False

# reset env
env.reset()

# run simulation
env_ids = torch.arange(env_unwrapped.num_envs, device=env_unwrapped.device)
et = rma_mdp._ensure_buffer(env_unwrapped, "rma_env_factors_buf", DEFAULT_ET_SPEC.dim)

# Sample a constant downward force (like mass attached to torso)
push_min, push_max = float(args_cli.push_range[0]), float(args_cli.push_range[1])
constant_force = rma_mdp.sample_payload_force(env_ids, env_unwrapped.device, (push_min, push_max))

print(f"[RMA] Starting simulation with constant downward force: {constant_force[0].item():.2f} N", flush=True)

for step_idx in range(args_cli.steps):
    if not simulation_app.is_running():
        break

    # Apply the same constant downward force every step
    fx = torch.zeros((env_unwrapped.num_envs, 1), device=env_unwrapped.device)
    fy = torch.zeros((env_unwrapped.num_envs, 1), device=env_unwrapped.device)
    fz = -constant_force.unsqueeze(-1)  # Shape: (num_envs, 1)

    # Forces shape: (num_envs, len(body_ids), 3) = (num_envs, 1, 3)
    forces = torch.stack((fx, fy, fz), dim=-1)
    torques = torch.zeros_like(forces)

    if step_idx == 0:
        print(f"[RMA] Force tensor shape: {forces.shape}", flush=True)
        print(f"[RMA] Force value: {forces[0]}", flush=True)

    # Set the wrench in the asset buffer (use local frame, don't flip is_global)
    try:
        robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            positions=None,
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=False,
        )
        if step_idx == 0:
            print("[RMA] set_external_force_and_torque SUCCESS", flush=True)
    except Exception as e:
        print(f"[RMA] set_external_force_and_torque FAILED: {e}", flush=True)

    # Store force magnitude in e_t[0]
    et[env_ids, 0] = constant_force

    # Write buffers BEFORE stepping (KEY POINT)
    try:
        robot.write_data_to_sim()
        if step_idx == 0:
            print("[RMA] write_data_to_sim SUCCESS", flush=True)
    except Exception as e:
        print(f"[RMA] write_data_to_sim FAILED: {e}", flush=True)

    # Step the environment
    step_out = env.step(zero_actions)

    if step_idx == 0 or step_idx == 1:
        # Readback actual applied force from asset data
        try:
            if hasattr(robot.data, "body_external_force_w"):
                applied_force = robot.data.body_external_force_w[0, 3]  # env 0, body 3 (torso)
                print(f"[RMA] step {step_idx} actual applied force: {applied_force}", flush=True)
        except Exception as e:
            print(f"[RMA] readback failed: {e}", flush=True)

print("[RMA] Simulation complete.", flush=True)

# close the simulator
env.close()
simulation_app.close()
