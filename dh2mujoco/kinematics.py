"""Elementary rotation / translation matrices and FK chain utilities."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Elementary 4×4 homogeneous transforms
# ---------------------------------------------------------------------------

def Rx(angle: float) -> np.ndarray:
    """Rotation about x-axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float
    )


def Ry(angle: float) -> np.ndarray:
    """Rotation about y-axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float
    )


def Rz(angle: float) -> np.ndarray:
    """Rotation about z-axis by *angle* radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )


def Tx(dist: float) -> np.ndarray:
    """Translation along x by *dist*."""
    T = np.eye(4)
    T[0, 3] = float(dist)
    return T


def Ty(dist: float) -> np.ndarray:
    """Translation along y by *dist*."""
    T = np.eye(4)
    T[1, 3] = float(dist)
    return T


def Tz(dist: float) -> np.ndarray:
    """Translation along z by *dist*."""
    T = np.eye(4)
    T[2, 3] = float(dist)
    return T


# ---------------------------------------------------------------------------
# DH convention transforms
# ---------------------------------------------------------------------------

def modified_dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Modified DH (Craig convention) single-link transform.

    T = Rx(alpha) · Tx(a) · Rz(theta) · Tz(d)

    Parameters match the "pre-frame" columns: a_{i-1}, alpha_{i-1}, d_i, theta_i.
    """
    return Rx(alpha) @ Tx(a) @ Rz(theta) @ Tz(d)


def standard_dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard DH single-link transform.

    T = Rz(theta) · Tz(d) · Tx(a) · Rx(alpha)

    Parameters: a_i, alpha_i, d_i, theta_i.
    """
    return Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha)


# ---------------------------------------------------------------------------
# FK helpers
# ---------------------------------------------------------------------------

def fk_chain(transforms: list[np.ndarray]) -> np.ndarray:
    """Left-to-right product of a list of 4×4 matrices."""
    T = np.eye(4, dtype=float)
    for Ti in transforms:
        T = T @ Ti
    return T


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation formula – arbitrary unit axis, 4×4 homogeneous."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    R = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=float,
    )
    T = np.eye(4)
    T[:3, :3] = R
    return T


def extract_position(T: np.ndarray) -> np.ndarray:
    """Return the (3,) translation vector from a 4×4 homogeneous matrix."""
    return T[:3, 3].copy()


def extract_rotation(T: np.ndarray) -> np.ndarray:
    """Return the (3,3) rotation matrix from a 4×4 homogeneous matrix."""
    return T[:3, :3].copy()
