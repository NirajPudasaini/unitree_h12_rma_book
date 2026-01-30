from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from rma_modules.env_factor_spec import DEFAULT_ET_SPEC, LEG_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def _ensure_buffer(env: ManagerBasedRLEnv, name: str, dim: int) -> torch.Tensor:
    buf = getattr(env, name, None)
    if buf is None or not isinstance(buf, torch.Tensor) or buf.shape != (env.num_envs, dim):
        buf = torch.zeros((env.num_envs, dim), device=env.device, dtype=torch.float)
        setattr(env, name, buf)
    return buf


def _resolve_asset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    # SceneEntityCfg in IsaacLab typically uses the attribute name as the key in env.scene.
    return env.scene[asset_cfg.name]


def _get_body_view(asset):
    view = None
    for candidate in ("body_physx_view", "link_physx_view", "_body_physx_view", "_link_physx_view"):
        if hasattr(asset, candidate):
            view = getattr(asset, candidate)
            break
    return view


def _read_leg_effort_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, env_ids: torch.Tensor, leg_joint_names: Sequence[str]
) -> tuple[torch.Tensor, list[int]] | tuple[None, None]:
    try:
        asset = _resolve_asset(env, asset_cfg)
        
        # Resolve joint IDs
        if hasattr(asset, "find_joints"):
            joint_ids, _ = asset.find_joints(list(leg_joint_names))
        
        if joint_ids is None:
            return None, None
        
        # Get and return limits
        data = getattr(asset, "data", None)
        if data is None or not hasattr(data, "joint_effort_limits"):
            return None, None
        
        limits = data.joint_effort_limits[env_ids][:, joint_ids]
        return limits, list(joint_ids)
    except Exception:
        return None, None


def _read_ground_friction(env: ManagerBasedRLEnv) -> float | None:
    try:
        # Try to access terrain
        terrain = getattr(env.scene, "terrain", None)
        if terrain is None:
            terrain = env.scene["terrain"]
        
        # Search for physics material in common locations
        for obj in (getattr(terrain, "cfg", None), terrain, getattr(terrain, "physics_material", None)):
            if obj is None:
                continue
            
            mat = getattr(obj, "physics_material", None)
            if mat is None:
                mat = obj if hasattr(obj, "static_friction") and hasattr(obj, "dynamic_friction") else None
            if mat is None:
                continue
            
            return float(getattr(mat, "static_friction"))
    except Exception:
        pass
    
    return None


def _apply_ground_friction(env: ManagerBasedRLEnv, friction_val: float) -> None:
    """Apply a single friction value to the terrain physics material (global)."""
    try:
        terrain = getattr(env.scene, "terrain", None)
        if terrain is None:
            terrain = env.scene["terrain"]

        # Search for physics material in common locations
        for obj in (getattr(terrain, "cfg", None), terrain, getattr(terrain, "physics_material", None)):
            if obj is None:
                continue

            mat = getattr(obj, "physics_material", None)
            if mat is None and hasattr(obj, "static_friction") and hasattr(obj, "dynamic_friction"):
                mat = obj
            if mat is None:
                continue

            # Set both static and dynamic friction
            if hasattr(mat, "static_friction"):
                setattr(mat, "static_friction", float(friction_val))
            if hasattr(mat, "dynamic_friction"):
                setattr(mat, "dynamic_friction", float(friction_val))
            return
    except Exception:
        pass



def _maybe_cache_baselines(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, env_ids: torch.Tensor) -> None:
    if not hasattr(env, "_rma_baseline_effort_limits"):
        limits, joint_ids = _read_leg_effort_limits(env, asset_cfg, env_ids, LEG_JOINT_NAMES)
        if limits is not None and joint_ids is not None:
            env._rma_baseline_effort_limits = limits.clone()
            env._rma_baseline_effort_joint_ids = joint_ids

    if not hasattr(env, "_rma_baseline_friction"):
        mu = _read_ground_friction(env)
        if mu is not None:
            env._rma_baseline_friction = mu


def sample_payload_force(
    env_ids: torch.Tensor,
    device: torch.device,
    force_range_n: tuple[float, float] = (0.0, 50.0),
) -> torch.Tensor:
    """Sample downward force magnitudes for each environment.
    
    Args:
        env_ids: environment indices being reset/sampled.
        device: torch device.
        force_range_n: (min_force, max_force) in Newtons.
    
    Returns:
        Tensor of shape (num_envs,) with sampled force magnitudes (in Newtons).
    """
    num = env_ids.numel()
    force_min, force_max = float(force_range_n[0]), float(force_range_n[1])
    force_samples = torch.empty((num,), device=device).uniform_(force_min, force_max)
    return force_samples


def sample_leg_strength_scale(
    env_ids: torch.Tensor,
    device: torch.device,
    num_joints: int = 12,
    strength_range: tuple[float, float] = (0.9, 1.1),
) -> torch.Tensor:
    """Sample leg strength scales for each joint.
    
    Args:
        env_ids: environment indices being reset/sampled.
        device: torch device.
        num_joints: number of leg joints (default 12 for Unitree-H12).
        strength_range: (min_scale, max_scale) e.g., (0.9, 1.1) for ±10%.
    
    Returns:
        Tensor of shape (num_envs, num_joints) with per-joint strength scales.
    """
    num = env_ids.numel()
    strength_min, strength_max = float(strength_range[0]), float(strength_range[1])
    strength_samples = torch.empty((num, num_joints), device=device).uniform_(strength_min, strength_max)
    return strength_samples


def _apply_downward_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
    force_n: torch.Tensor,
) -> None:
    """Best-effort: apply downward external force to torso body."""

    try:
        asset = _resolve_asset(env, asset_cfg)
    except Exception:
        return

    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None:
        return
    if isinstance(body_ids, int):
        body_ids = [body_ids]

    data = getattr(asset, "data", None)
    if data is None or not hasattr(data, "body_external_force_w"):
        return

    try:
        forces = data.body_external_force_w
        forces[env_ids[:, None], torch.tensor(body_ids, device=env.device)] = 0.0
        forces[env_ids[:, None], torch.tensor(body_ids, device=env.device), 2] = -force_n.unsqueeze(-1)
        if hasattr(asset, "write_body_external_force_to_sim"):
            asset.write_body_external_force_to_sim(forces)
        elif hasattr(asset, "write_data_to_sim"):
            asset.write_data_to_sim()
    except Exception:
        pass


def _set_leg_effort_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
    leg_strength_scale: torch.Tensor,
    leg_joint_names: Sequence[str],
) -> None:
    """Best-effort: scale per-joint effort/torque limits for leg joints.

    Different IsaacLab versions expose different access patterns; we try a few common ones.
    If we cannot apply, we silently no-op (but still store e_t).
    """

    try:
        asset = _resolve_asset(env, asset_cfg)
    except Exception:
        return

    # Resolve joint ids
    joint_ids = None
    try:
        if hasattr(asset, "find_joints"):
            joint_ids, _ = asset.find_joints(list(leg_joint_names))
        elif hasattr(asset, "find_joints_by_name"):
            joint_ids = asset.find_joints_by_name(list(leg_joint_names))
    except Exception:
        joint_ids = None

    if joint_ids is None:
        return

    # Try to access joint effort limits
    try:
        data = getattr(asset, "data", None)
        if data is not None and hasattr(data, "joint_effort_limits"):
            limits = data.joint_effort_limits
            limits_env = limits[env_ids][:, joint_ids]
            limits_env = limits_env * leg_strength_scale
            limits[env_ids[:, None], torch.tensor(joint_ids, device=env.device)] = limits_env
            if hasattr(asset, "write_joint_effort_limits_to_sim"):
                asset.write_joint_effort_limits_to_sim(limits)
                #print after writing to verify
            return
    except Exception:
        pass



def sample_rma_env_factors(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    *,
    payload_force_range_n: tuple[float, float] = (0.0, 50.0),
    leg_strength_range: tuple[float, float] = (0.9, 1.1),
    friction_range: tuple[float, float] = (0.5, 1.0),
    apply_to_sim: bool = True,
) -> None:
    """Sample + store + apply e_t (14D: force + leg_strength + friction).

    Active factors (sampled & applied):
        - e_t[0] (1D): downward force in Newtons → applied as external wrench
        - e_t[1:13] (12D): leg joint strength scales → applied as torque limit scaling
        - e_t[13] (1D): ground friction coefficient (sampled, observed by encoder)

    Args:
        env_ids: environments being reset.
        asset_cfg: Scene entity (robot).
        apply_to_sim: if True, applies force and leg strength to simulator.
    """

    device = env.device
    env_ids = env_ids.to(device=device)

    et = _ensure_buffer(env, "rma_env_factors_buf", DEFAULT_ET_SPEC.dim)

    # --- sample active factors (e_t = force + leg_strength + friction, 14D per RmaEtSpec)
    payload_force = sample_payload_force(env_ids, device, payload_force_range_n)
    leg_strength = sample_leg_strength_scale(env_ids, device, DEFAULT_ET_SPEC.leg_strength_dim, leg_strength_range)

    # --- sample friction globally (single value for all envs)
    friction_val = getattr(env, "_rma_curriculum_friction", None)
    if friction_val is None:
        levels = getattr(env, "_rma_friction_levels", None)
        if levels is not None and len(levels) > 0:
            friction_val = float(levels[0].item())
        else:
            mu_min, mu_max = float(friction_range[0]), float(friction_range[1])
            friction_val = torch.empty((1,), device=device).uniform_(mu_min, mu_max).item()
    else:
        friction_val = float(friction_val)
    friction = torch.full((env_ids.numel(),), friction_val, device=device)

    # --- pack e_t buffer (14D: payload, leg_strength, friction)
    et_env = et[env_ids]
    et_env[:, DEFAULT_ET_SPEC.payload_slice] = payload_force.unsqueeze(-1)
    et_env[:, DEFAULT_ET_SPEC.leg_strength_slice] = leg_strength
    et_env[:, DEFAULT_ET_SPEC.friction_slice] = friction.unsqueeze(-1)
    et[env_ids] = et_env

    # Cache baselines before any modifications
    env._rma_asset_cfg = asset_cfg
    _maybe_cache_baselines(env, asset_cfg, env_ids)

    # --- apply to simulator
    if apply_to_sim:
        _apply_downward_force(env, asset_cfg, env_ids, payload_force)
        _set_leg_effort_limits(env, asset_cfg, env_ids, leg_strength, LEG_JOINT_NAMES)
        # Terrain friction is global; apply the sampled value
        _apply_ground_friction(env, float(friction_val))

    # Store for debugging
    env.rma_payload_force_n = payload_force
    env.rma_leg_strength_scale = leg_strength
