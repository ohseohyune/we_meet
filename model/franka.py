"""
6-DOF Standard-DH PoE kinematic model.

The module name is kept for backward compatibility with the existing examples.
Kinematic parameters come from the provided 6-DOF Standard DH XML:

    A_i = Rz(theta_offset_i + q_i) Tz(d_i) Tx(a_i) Rx(alpha_i)

The 7th DH row is a fixed tool/flange transform.
"""

from __future__ import annotations

import numpy as np

from robot.kinematics import T_inv, adjoint


Q_MIN = np.full(6, -np.pi)
Q_MAX = np.full(6, np.pi)
HOME_Q = np.zeros(6)

# Rows are [d, theta_offset, a, alpha].
DH_PARAMS = np.array(
    [
        [0.132184, 3.146991, -0.061099, -1.568837],
        [0.091502, 0.002344, 0.345932, -0.004401],
        [0.005000, -1.577745, 0.103901, 1.574759],
        [0.344628, -0.009289, -0.000588, -1.581564],
        [-0.024818, -3.104470, 0.074472, -1.553936],
        [0.042524, 1.581094, 0.070882, -1.520796],
    ],
    dtype=float,
)

TOOL_D = -0.2445
TOOL_THETA_OFFSET = -np.pi / 2.0
TOOL_A = 0.0
TOOL_ALPHA = 0.0


def dh_transform(d: float, theta: float, a: float, alpha: float) -> np.ndarray:
    """Standard DH transform Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _space_screw_from_joint_frame(T_base_joint: np.ndarray) -> np.ndarray:
    """Revolute screw [omega; v] for a joint whose axis is local +z."""
    omega = T_base_joint[:3, 2].copy()
    q = T_base_joint[:3, 3].copy()
    v = -np.cross(omega, q)
    return np.concatenate([omega, v])


def build_dh6_poe_model(
    dh_params: np.ndarray = DH_PARAMS,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Convert the 6-DOF Standard DH chain to Space/Body PoE parameters.

    Returns
    -------
    S_list
        Space screw axes at the zero joint configuration.
    M
        Home pose of the TCP/flange frame.
    B_list
        Body screw axes satisfying T(q)=M exp(B1 q1)...exp(B6 q6).
    """
    T = np.eye(4)
    S_list: list[np.ndarray] = []

    for d, theta_offset, a, alpha in np.asarray(dh_params, dtype=float):
        S_list.append(_space_screw_from_joint_frame(T))
        T = T @ dh_transform(d, theta_offset, a, alpha)

    M = T @ dh_transform(TOOL_D, TOOL_THETA_OFFSET, TOOL_A, TOOL_ALPHA)
    B_list = [adjoint(T_inv(M)) @ S for S in S_list]
    return S_list, M, B_list


S_LIST, M, B_LIST = build_dh6_poe_model()
