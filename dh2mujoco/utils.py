"""Formatting and printing utilities."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .quaternion import rotation_matrix_to_quat, rotation_matrix_to_euler_xyz


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def fmt(value: float, precision: int = 9) -> str:
    """Format a float with *precision* significant figures (no trailing zeros)."""
    return f"{float(value):.{precision}g}"


def vec(values, precision: int = 9) -> str:
    """Format a sequence of floats as a space-separated string."""
    return " ".join(fmt(v, precision) for v in values)


# ---------------------------------------------------------------------------
# Transform printing
# ---------------------------------------------------------------------------

def print_transform(
    label: str,
    T: np.ndarray,
    show_matrix: bool = True,
    show_quaternion: bool = True,
) -> None:
    """Pretty-print a 4×4 homogeneous transform."""
    pos = T[:3, 3]
    R = T[:3, :3]
    q = rotation_matrix_to_quat(R)
    euler = rotation_matrix_to_euler_xyz(R)

    sep = "-" * 56
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Position  : [{pos[0]:+.6f}  {pos[1]:+.6f}  {pos[2]:+.6f}] m")
    if show_quaternion:
        print(
            f"  Quaternion: w={q[0]:+.6f}  x={q[1]:+.6f}  "
            f"y={q[2]:+.6f}  z={q[3]:+.6f}"
        )
    print(
        f"  Euler XYZ : roll={np.degrees(euler[0]):+.3f}°  "
        f"pitch={np.degrees(euler[1]):+.3f}°  "
        f"yaw={np.degrees(euler[2]):+.3f}°"
    )
    if show_matrix:
        print("  Rotation matrix:")
        for row in R:
            print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
    print(f"  Homogeneous transform:")
    for row in T:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}  {row[3]:+.6f}]")


def print_fk_chain(
    chain: List[np.ndarray],
    n_joints: int,
    show_matrix: bool = False,
) -> None:
    """Print every frame in a FK chain."""
    for i, T in enumerate(chain):
        if i == 0:
            label = "World origin (T_0)"
        elif i <= n_joints:
            label = f"Joint {i} frame (T_{i})"
        else:
            ee_idx = i - n_joints
            label = f"EE sub-frame {ee_idx} (T_{i})"
        print_transform(label, T, show_matrix=show_matrix)
