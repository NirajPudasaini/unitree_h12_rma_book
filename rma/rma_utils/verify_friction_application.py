"""Standalone script to verify friction sampling and application."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add IsaacLab + local extensions to PYTHONPATH
workspace_root = Path(__file__).resolve().parents[2]
isaaclab_root = Path("/home/niraj/isaac_projects/IsaacLab")
sys.path.insert(0, str(isaaclab_root / "source"))
sys.path.insert(0, str(workspace_root / "isaaclab" / "source"))
sys.path.insert(0, str(workspace_root / "rma"))
sys.path.insert(0, str(workspace_root / "isaaclab" / "scripts" / "rsl_rl"))

# NOTE: Do not import any Isaac Sim/Omni modules before SimulationApp starts.
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Verify friction sampling and application in sim.")
parser.add_argument("--task", type=str, default="Unitree-H12-Walk-RMA-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment.")
parser.add_argument("--num_resets", type=int, default=3, help="Number of resets to inspect.")
parser.add_argument("--manual_sample", action="store_true", help="Manually resample friction via rma_mdp.")
parser.add_argument("--friction_min", type=float, default=0.5, help="Min friction for manual sampling.")
parser.add_argument("--friction_max", type=float, default=1.0, help="Max friction for manual sampling.")
parser.add_argument("--apply_to_sim", action="store_true", default=True, help="Apply manual samples to sim.")
parser.add_argument(
    "--use_existing_kit_app",
    action="store_true",
    default=False,
    help="Skip AppLauncher (use when running via `isaacsim --exec`).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app (or reuse existing Kit app)
simulation_app = None
existing_kit_app = False
try:
    import omni.kit_app  # type: ignore

    existing_kit_app = omni.kit_app.get_app() is not None
except Exception:
    existing_kit_app = False

use_existing_kit_app = (
    args_cli.use_existing_kit_app
    or existing_kit_app
    or os.environ.get("OMNI_APP_NAME")
    or os.environ.get("ISAACSIM_APP")
    or os.environ.get("OMNI_KIT_APP")
)
if not use_existing_kit_app:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

# Ensure environment registration is executed after SimulationApp starts
import unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim  # noqa: F401

import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from rma_modules.env_factor_spec import DEFAULT_ET_SPEC
from unitree_h12_sim2sim.tasks.manager_based.unitree_h12_sim2sim.mdp import rma as rma_mdp


def _read_friction_from_buffer(unwrapped) -> tuple[float | None, float | None, float | None]:
    buf = getattr(unwrapped, "rma_env_factors_buf", None)
    if buf is None or buf.numel() == 0:
        return None, None, None
    friction = buf[:, DEFAULT_ET_SPEC.friction_slice].squeeze(-1)
    return float(friction.min().item()), float(friction.max().item()), float(friction.mean().item())


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, _agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    unwrapped = env.unwrapped
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    asset_cfg = SceneEntityCfg("robot")

    print("[INFO] Starting friction verification...")
    print(f"[INFO] Task: {args_cli.task} | Num envs: {unwrapped.num_envs} | Device: {unwrapped.device}")

    for i in range(args_cli.num_resets):
        env.reset()

        et_min, et_max, et_mean = _read_friction_from_buffer(unwrapped)
        sim_mu = rma_mdp._read_ground_friction(unwrapped)
        print(f"[RESET {i}] e_t friction min/max/mean: {et_min}/{et_max}/{et_mean} | sim friction: {sim_mu}")

        if args_cli.manual_sample:
            rma_mdp.sample_rma_env_factors(
                unwrapped,
                env_ids,
                asset_cfg,
                friction_range=(args_cli.friction_min, args_cli.friction_max),
                apply_to_sim=args_cli.apply_to_sim,
            )
            et_min, et_max, et_mean = _read_friction_from_buffer(unwrapped)
            sim_mu = rma_mdp._read_ground_friction(unwrapped)
            print(
                f"[MANUAL {i}] e_t friction min/max/mean: {et_min}/{et_max}/{et_mean} | sim friction: {sim_mu}"
            )

    env.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
