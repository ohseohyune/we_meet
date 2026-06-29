"""Quaternion utilities.

Convention throughout: (w, x, y, z)  – matching MuJoCo's attribute order.
"""

from __future__ import annotations

import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_rx(alpha: float) -> np.ndarray:
    """Unit quaternion for rotation by *alpha* radians about x."""
    h = 0.5 * float(alpha)
    return np.array([np.cos(h), np.sin(h), 0.0, 0.0])


def quat_ry(beta: float) -> np.ndarray:
    """Unit quaternion for rotation by *beta* radians about y."""
    h = 0.5 * float(beta)
    return np.array([np.cos(h), 0.0, np.sin(h), 0.0])


def quat_rz(theta: float) -> np.ndarray:
    """Unit quaternion for rotation by *theta* radians about z."""
    h = 0.5 * float(theta)
    return np.array([np.cos(h), 0.0, 0.0, np.sin(h)])


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Shepperd's method: 3×3 rotation matrix → unit quaternion (w,x,y,z).

    Always returns a quaternion with w ≥ 0 (canonical form).
    """
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) → 3×3 rotation matrix."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → (roll, pitch, yaw) in radians (XYZ intrinsic)."""
    R = np.asarray(R, dtype=float)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=float)


def quat_angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle [rad] between two unit quaternions."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    dot = abs(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)))
    dot = min(dot, 1.0)
    return 2.0 * np.arccos(dot)
