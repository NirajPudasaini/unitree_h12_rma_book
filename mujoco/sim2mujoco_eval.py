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

"""Main entry point for sim2mujoco evaluation."""

import argparse
from pathlib import Path

import numpy as np
import torch

from sim2mujoco.actions import ActionProcessor
from sim2mujoco.commands import CommandManager
from sim2mujoco.observations import ObservationProcessor
from sim2mujoco.policy import PolicyWrapper
from sim2mujoco.simulation import MuJocoSimulation
from sim2mujoco.utils import default_device, load_config


def main():
    """Run sim2sim evaluation."""
    parser = argparse.ArgumentParser(description="Sim2Sim Policy Evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to policy checkpoint (.pt or .onnx)")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--mjcf", type=Path, default=None, help="Path to MJCF file (optional, overrides config)")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (seconds)")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo viewer")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency (control steps)")
    parser.add_argument(
        "--pd-scale", type=float, default=1.0, help="Scale factor for PD gains (use 0.3-0.5 for stability)"
    )
    parser.add_argument(
        "--disable-keyboard", action="store_true", help="Disable keyboard control for interactive commands"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable step-by-step logging output")
    parser.add_argument("--vel-x", type=float, default=0.0, help="Linear velocity X command")
    parser.add_argument("--vel-y", type=float, default=0.0, help="Linear velocity Y command")
    parser.add_argument("--ang-z", type=float, default=0.0, help="Angular velocity Z command")

    args = parser.parse_args()

    # Setup device.
    if args.device == "auto":
        device = default_device()
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load config.
    print(f"\nLoading config from {args.config}...")
    config = load_config(args.config)

    # Override MJCF path if provided.
    if args.mjcf:
        config["mjcf_path"] = str(args.mjcf)

    # Scale PD gains if requested.
    if args.pd_scale != 1.0:
        print(f"Scaling PD gains by {args.pd_scale}...")
        robot_config = config["articulations"]["robot"]
        robot_config["default_joint_stiffness"] = [kp * args.pd_scale for kp in robot_config["default_joint_stiffness"]]
        robot_config["default_joint_damping"] = [kd * args.pd_scale for kd in robot_config["default_joint_damping"]]

    # Create command manager if keyboard control is enabled.
    command_manager = CommandManager(
        device=device, defaults={"linear_x": args.vel_x, "linear_y": args.vel_y, "angular_z": args.ang_z}
    )
    if not args.disable_keyboard and not args.no_viewer:
        print("\n✓ Keyboard control enabled")
    else:
        print("\n✓ Keyboard control disabled (using fixed commands)")

    # Load policy.
    print(f"\nLoading policy from {args.checkpoint}...")
    policy = PolicyWrapper.from_config(args.checkpoint, config, device)
    print(f"  Policy type: {type(policy).__name__}")

    # Create simulation.
    print("\nCreating simulation...")
    sim = MuJocoSimulation(
        config, device, enable_viewer=not args.no_viewer, mjcf_path=args.mjcf, command_manager=command_manager
    )
    print(f"  Num joints: {sim.num_joints}")
    print(f"  Fixed base: {sim.fixed_base}")
    print(f"  Physics dt: {sim.physics_dt}s ({1.0 / sim.physics_dt:.0f} Hz)")
    print(f"  Control dt: {sim.dt}s ({1.0 / sim.dt:.1f} Hz)")
    print(f"  Decimation: {sim.decimation}")

    # Create processors.
    print("\nSetting up observation processor...")
    obs_processor = ObservationProcessor(config, sim.joint_names, device, command_manager=command_manager)
    print(f"  Total observation dim: {obs_processor.total_obs_dim}")
    print("  Observation terms:")
    for term in obs_processor.terms:
        hist_info = f" (history={term.history_length})" if term.history_length > 0 else ""
        print(f"    - {term.name}: {term.output_dim()}{hist_info}")

    print("\nSetting up action processor...")
    act_processor = ActionProcessor(config, sim.joint_names, device)
    print(f"  Total action dim: {act_processor.total_action_dim}")
    print("  Action terms:")
    for term in act_processor.action_terms:
        print(f"    - {term.name}: {term.action_dim} joints (scale: {term.scale})")

    # Reset.
    sim.reset()
    obs_processor.reset()
    policy.reset()

    # Evaluation loop.
    control_dt = sim.dt  # This is physics_dt * decimation
    physics_dt = sim.physics_dt
    num_steps = int(args.duration / control_dt)

    print(f"\nRunning evaluation for {args.duration}s ({num_steps} control steps)...")
    print(f"  Control frequency: {1.0 / control_dt:.1f} Hz")
    print(f"  Physics frequency: {1.0 / physics_dt:.1f} Hz")
    print("-" * 80)

    # Printing setup
    print_interval = 100
    max_prints = 3
    print_count = 0

    try:
        for step in range(num_steps):
            # Get observations.
            sim_state = sim.get_state()
            obs = obs_processor.compute(sim_state)

            # Policy inference.
            with torch.no_grad():
                actions = policy(obs)

            # Print observations and actions at intervals
            if step % print_interval == 0 and print_count < max_prints:
                print(f"\n{'='*100}")
                print(f"[STEP {step}] - Observation & Action Snapshot (Print {print_count + 1}/{max_prints})")
                print(f"{'='*100}")
                
                # Print observations
                print(f"[OBSERVATIONS] shape={obs.shape}, dtype={obs.dtype}")
                obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
                obs_formatted = np.array2string(obs_np, formatter={'float_kind': lambda x: f'{x:.2f}'}, separator=', ')
                print(f"  Obs (all {len(obs_np)} dims):\n  {obs_formatted}")
                
                # Print actions
                print(f"\n[ACTIONS] shape={actions.shape}, dtype={actions.dtype}")
                actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else actions
                actions_formatted = np.array2string(actions_np, formatter={'float_kind': lambda x: f'{x:.2f}'}, separator=', ')
                print(f"  Actions: {actions_formatted}")
                print(f"  Overall - Min: {actions.min().item():.4f}, Max: {actions.max().item():.4f}, Mean: {actions.mean().item():.4f}")
                print(f"{'='*100}\n")
                print_count += 1

            # Update last action.
            obs_processor.set_last_action(actions)

            # Process actions.
            joint_cmd = act_processor.process(actions)

            # Step simulation (decimation times).
            for _ in range(sim.decimation):
                sim.step(joint_cmd)

            # Logging (get fresh state AFTER simulation steps).
            if args.verbose and step % args.log_freq == 0:
                current_state = sim.get_state()
                print(
                    f"Step {step:4d}/{num_steps} | "
                    f"Root pos: [{current_state.root_pos[0]:6.3f}, {current_state.root_pos[1]:6.3f}, {current_state.root_pos[2]:6.3f}] | "
                    f"Root vel: [{current_state.root_lin_vel[0]:6.3f}, {current_state.root_lin_vel[1]:6.3f}, {current_state.root_lin_vel[2]:6.3f}] | "
                    f"Action mean: {actions.mean().item():7.4f}, std: {actions.std().item():7.4f}"
                )

        print("-" * 80)
        print(f"\nEvaluation complete! Ran {num_steps} steps.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        sim.close()
        print("Simulation closed.")


if __name__ == "__main__":
    main()
