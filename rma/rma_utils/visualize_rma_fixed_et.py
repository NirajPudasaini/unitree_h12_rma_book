#!/usr/bin/env python3
"""Visualize RMA effects with fixed e_t values (no sampling).

This script tests the RMA module with constant environment factors to observe
how force, leg strength, friction, and terrain parameters affect the robot's gait.

Usage:
    python visualize_rma_fixed_et.py --force 25 --strength 1.0 --friction 1.0 --num-envs 2 --duration 10
"""

import argparse
import sys
from pathlib import Path
import torch

# Add paths - the package is installed as editable, but ensure both levels are available
proj_root = Path(__file__).parent.parent
sys.path.insert(0, str(proj_root / "isaaclab" / "source"))
sys.path.insert(0, str(proj_root / "isaaclab" / "source" / "unitree_h12_sim2sim"))

from isaaclab.app import AppLauncher

# Launch the app
app_launcher = AppLauncher(headless=False)
sim = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def main(args):
    """Run visualization with fixed e_t."""
    
    # Import after path is set and app is launched
    from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim import (
        unitree_h12_walk_rma_cfg,
    )
    from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim.mdp.rma import verify_rma_env_factors_once
    from rma_modules.env_factor_spec import DEFAULT_ET_SPEC, LEG_JOINT_NAMES
    
    # Create environment
    env_cfg = unitree_h12_walk_rma_cfg.H12LocomotionFullBodyRmaEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"\n{'='*80}")
    print(f"RMA Fixed E_t Visualization")
    print(f"{'='*80}")
    print(f"Force: {args.force:.2f} N")
    print(f"Leg Strength: {args.strength:.2f}")
    print(f"Friction: {args.friction:.2f}")
    print(f"Num Envs: {args.num_envs}")
    print(f"Duration: {args.duration:.2f}s")
    print(f"{'='*80}\n")
    
    # Initialize environment
    obs, _ = env.reset()
    
    # Get asset config for RMA application
    asset_cfg = SceneEntityCfg("robot", body_names="torso_link")
    
    # Helper to apply fixed e_t
    def apply_fixed_et():
        """Apply fixed e_t values to all environments."""
        device = env.device
        env_ids = torch.arange(env.num_envs, device=device)
        
        # Create fixed e_t tensors (no sampling, all constant)
        payload_force = torch.full((env.num_envs,), float(args.force), device=device)
        
        leg_strength = torch.full(
            (env.num_envs, 12),
            float(args.strength),
            device=device
        )
        
        friction = torch.full((env.num_envs,), float(args.friction), device=device)
        
        # Read terrain encoding (curriculum will control this)
        from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim.mdp.rma import _terrain_encoding_from_env
        terrain = _terrain_encoding_from_env(env, env_ids)
        
        # Manually pack e_t buffer
        et = getattr(env, "rma_env_factors_buf", None)
        if et is None:
            et = torch.zeros((env.num_envs, DEFAULT_ET_SPEC.dim), device=device, dtype=torch.float)
            env.rma_env_factors_buf = et
        
        et[:, DEFAULT_ET_SPEC.payload_slice] = payload_force.unsqueeze(-1)
        et[:, DEFAULT_ET_SPEC.leg_strength_slice] = leg_strength
        et[:, DEFAULT_ET_SPEC.friction_slice] = friction.unsqueeze(-1)
        et[:, DEFAULT_ET_SPEC.terrain_slice] = terrain
        
        # Cache baselines
        from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim.mdp.rma import _maybe_cache_baselines
        env._rma_asset_cfg = asset_cfg
        _maybe_cache_baselines(env, asset_cfg, env_ids)
        
        # Apply to simulator (force + leg strength)
        from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim.mdp.rma import _apply_downward_force, _set_leg_effort_limits
        _apply_downward_force(env, asset_cfg, env_ids, payload_force)
        _set_leg_effort_limits(env, asset_cfg, env_ids, leg_strength, LEG_JOINT_NAMES)
        
        # Store for debugging
        env.rma_payload_force_n = payload_force
        env.rma_leg_strength_scale = leg_strength
    
    # Reset and apply fixed e_t
    obs, _ = env.reset()
    env_ids = torch.arange(min(args.num_envs, env.num_envs), device=env.device)
    apply_fixed_et()
    
    # Verify once
    verify_rma_env_factors_once(
        env,
        env_ids,
        max_envs=min(2, args.num_envs),
        prefix="[RMA:fixed_et]"
    )
    
    # Run visualization loop
    max_steps = int(args.duration * 50)  # 50 Hz control rate
    print(f"Running for {max_steps} steps ({args.duration:.2f}s)...\n")
    
    # Get action dimension from environment
    action_dim = env.action_manager.total_action_dim
    print(f"Action dimension: {action_dim}\n")
    
    for step in range(max_steps):
        with torch.no_grad():
            # Generate actions: zeros for all joints (passive locomotion under disturbance)
            actions = torch.zeros((env.num_envs, action_dim), device=env.device)
            
            step_result = env.step(actions)
            obs = step_result[0]
        
        # Re-apply fixed e_t every N steps to ensure persistence
        if step % 100 == 0 and step > 0:
            apply_fixed_et()
    
    print(f"\n{'='*80}")
    print("Visualization complete!")
    print(f"{'='*80}\n")
    
    env.close()
    app_launcher.app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RMA effects with fixed e_t")
    parser.add_argument(
        "--force",
        type=float,
        default=0.0,
        help="Fixed downward force in Newtons (0-50 N)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Fixed leg strength multiplier (0.9-1.1)"
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="Fixed ground friction coefficient"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2,
        help="Number of environments to simulate"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds"
    )
    
    args = parser.parse_args()
    main(args)
