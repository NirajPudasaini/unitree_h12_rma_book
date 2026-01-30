# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401

# Explicitly import the Unitree H12 stand config to register the environment
from . import unitree_h12_sim2sim  # noqa: F401
