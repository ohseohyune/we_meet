"""
Franka Emika Panda PoE kinematic model.

The screw axes are generated from the manufacturer-style modified DH table:
    A_i = Rx(alpha_i) Tx(a_i) Rz(theta_i) Tz(d_i)

Only kinematics are modeled here. Dynamics, collision geometry, gripper fingers,
and MuJoCo mesh frames are intentionally outside this module.
"""

from __future__ import annotations

import numpy as np

from robot.kinematics import adjoint, T_inv


# Franka Panda joint limits from the FCI control-parameter documentation.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# Comfortable non-singular seed used by the examples.
HOME_Q = np.array([0.0, -0.45, 0.0, -2.25, 0.0, 2.05, 0.75])

# Rows are [a, alpha, d, theta_offset] for joints 1..7.
# A fixed flange transform is appended after joint 7.
DH_PARAMS = np.array(
    [
        [0.0, 0.0, 0.333, 0.0],
        [0.0, -np.pi / 2.0, 0.0, 0.0],
        [0.0, np.pi / 2.0, 0.316, 0.0],
        [0.0825, np.pi / 2.0, 0.0, 0.0],
        [-0.0825, -np.pi / 2.0, 0.384, 0.0],
        [0.0, np.pi / 2.0, 0.0, 0.0],
        [0.088, np.pi / 2.0, 0.0, 0.0],
    ],
    dtype=float,
)

FLANGE_D = 0.107
FLANGE_THETA_OFFSET = -np.pi / 4.0


def _rx_tx(a: float, alpha: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [1.0, 0.0, 0.0, a],
            [0.0, ca, -sa, 0.0],
            [0.0, sa, ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _rz_tz(theta: float, d: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, 0.0],
            [st, ct, 0.0, 0.0],
            [0.0, 0.0, 1.0, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Modified DH transform Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    return _rx_tx(a, alpha) @ _rz_tz(theta, d)


def _space_screw_from_joint_frame(T_base_joint: np.ndarray) -> np.ndarray:
    """Revolute screw [omega; v] from the current DH joint frame."""
    omega = T_base_joint[:3, 2].copy()
    q = T_base_joint[:3, 3].copy()
    v = -np.cross(omega, q)
    return np.concatenate([omega, v])


def build_franka_poe_model(
    dh_params: np.ndarray = DH_PARAMS,
    flange_d: float = FLANGE_D,
    flange_theta_offset: float = FLANGE_THETA_OFFSET,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """
    Convert the Franka DH chain to Space/Body PoE parameters.

    Returns
    -------
    S_list
        Space screw axes at the zero joint configuration.
    M
        Home pose of the flange/TCP frame.
    B_list
        Body screw axes satisfying T(q)=M exp(B1 q1)...exp(B7 q7).
    """
    T = np.eye(4)
    S_list: list[np.ndarray] = []

    for a, alpha, d, theta_offset in np.asarray(dh_params, dtype=float):
        T_joint = T @ _rx_tx(a, alpha)
        S_list.append(_space_screw_from_joint_frame(T_joint))
        T = T_joint @ _rz_tz(theta_offset, d)

    M = T @ dh_transform(0.0, 0.0, flange_d, flange_theta_offset)
    B_list = [adjoint(T_inv(M)) @ S for S in S_list]
    return S_list, M, B_list


S_LIST, M, B_LIST = build_franka_poe_model()
