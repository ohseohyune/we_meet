"""ik_solver.py - Public entry point for the MuJoCo IK modules.

The solver implementation is split by role:
  arm_model.py      joint/site utilities and collision checks
  look_at_ik.py     single-pose look-at IK solvers
  trajectory_ik.py  trajectory-level solving (DP path optimizer), evaluation,
                    motion metrics, and retiming

Existing imports (`from control.ik_solver import ...`) keep working through
the re-exports below.
"""

from __future__ import annotations

from .arm_model import (
    ARM_JOINT_NAMES,
    ROBOT_COLLISION_BIT,
    collision_contacts_for_q,
    evaluate_collision_trajectory,
    get_arm_qpos,
    joint_limits,
    joint_qpos_indices,
    set_arm_qpos,
    site_pose,
)
from .look_at_ik import (
    solve_look_at_ik,
    solve_look_at_ik_optimized,
)
from .trajectory_ik import (
    evaluate_look_at_trajectory,
    joint_motion_metrics,
    retime_joint_trajectory,
    solve_trajectory,
)

__all__ = [
    "ARM_JOINT_NAMES",
    "ROBOT_COLLISION_BIT",
    "collision_contacts_for_q",
    "evaluate_collision_trajectory",
    "evaluate_look_at_trajectory",
    "get_arm_qpos",
    "joint_limits",
    "joint_motion_metrics",
    "joint_qpos_indices",
    "retime_joint_trajectory",
    "set_arm_qpos",
    "site_pose",
    "solve_look_at_ik",
    "solve_look_at_ik_optimized",
    "solve_trajectory",
]
