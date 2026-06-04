"""
Body-Jacobian CLIK for the 7-DOF Franka Panda.
"""

from __future__ import annotations

import numpy as np

from control.jacobian import body_jacobian
from robot.kinematics import T_inv, body_poe_fk
from utils.lie_group import matrix_log_se3


def compute_body_error(T_cur: np.ndarray, T_des: np.ndarray) -> np.ndarray:
    """Return body-frame SE(3) error twist from current pose to desired pose."""
    return matrix_log_se3(T_inv(T_cur) @ T_des)


def damped_pinv(J: np.ndarray, damping: float = 0.05) -> np.ndarray:
    """Damped least-squares pseudo-inverse."""
    J = np.asarray(J, dtype=float)
    if damping <= 0.0:
        return np.linalg.pinv(J)
    eye = np.eye(J.shape[0])
    return J.T @ np.linalg.inv(J @ J.T + damping**2 * eye)


def joint_limit_avoidance_velocity(
    theta: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    gain: float = 0.08,
) -> np.ndarray:
    """
    Null-space preference velocity that pulls joints toward the limit midpoint.

    This is intentionally conservative; the hard safety action is still clipping
    after integration in clik_step().
    """
    theta = np.asarray(theta, dtype=float)
    q_lo = np.asarray(q_lo, dtype=float)
    q_hi = np.asarray(q_hi, dtype=float)
    q_mid = 0.5 * (q_lo + q_hi)
    q_range = np.maximum(q_hi - q_lo, 1e-9)
    return gain * (q_mid - theta) / q_range


def clik_step(
    theta: np.ndarray,
    T_des: np.ndarray,
    B_list: list[np.ndarray],
    M: np.ndarray,
    K_p: np.ndarray,
    dt: float,
    damping: float = 0.05,
    q_lo: np.ndarray | None = None,
    q_hi: np.ndarray | None = None,
    theta_dot0: np.ndarray | None = None,
    nullspace_gain: float = 0.08,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    One resolved-rate CLIK step with 7-DOF null-space projection.

        qdot = Jb^+ Kp eb + (I - Jb^+ Jb) qdot0
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    T_cur = body_poe_fk(theta, B_list, M)
    e_b = compute_body_error(T_cur, T_des)
    Jb = body_jacobian(theta, B_list)
    control = np.asarray(K_p, dtype=float).reshape(6, 6) @ e_b

    J_pinv = damped_pinv(Jb, damping=damping)

    if theta_dot0 is None:
        if q_lo is not None and q_hi is not None:
            theta_dot0 = joint_limit_avoidance_velocity(
                theta, q_lo, q_hi, gain=nullspace_gain
            )
        else:
            theta_dot0 = np.zeros_like(theta)
    else:
        theta_dot0 = np.asarray(theta_dot0, dtype=float).reshape(theta.shape)

    null_projector = np.eye(theta.size) - J_pinv @ Jb
    theta_dot = J_pinv @ control + null_projector @ theta_dot0
    theta_new = theta + dt * theta_dot

    if q_lo is not None or q_hi is not None:
        lo = -np.inf if q_lo is None else np.asarray(q_lo, dtype=float)
        hi = np.inf if q_hi is None else np.asarray(q_hi, dtype=float)
        theta_new = np.clip(theta_new, lo, hi)

    if not return_info:
        return theta_new

    info = {
        "T_cur": T_cur,
        "error": e_b,
        "error_norm": float(np.linalg.norm(e_b)),
        "Jb": Jb,
        "condition": float(np.linalg.cond(Jb)),
        "theta_dot": theta_dot,
        "theta_dot0": theta_dot0,
    }
    return theta_new, info


def solve_ik(
    T_des: np.ndarray,
    B_list: list[np.ndarray],
    M: np.ndarray,
    q_init: np.ndarray,
    K_p: np.ndarray | None = None,
    max_iter: int = 300,
    tol: float = 1e-4,
    dt: float = 0.04,
    damping: float = 0.05,
    q_lo: np.ndarray | None = None,
    q_hi: np.ndarray | None = None,
    nullspace_gain: float = 0.08,
) -> tuple[np.ndarray, bool, dict]:
    """Iterative Body-Jacobian IK with Franka-friendly redundancy handling."""
    q = np.asarray(q_init, dtype=float).reshape(-1).copy()
    if q_lo is not None or q_hi is not None:
        lo = -np.inf if q_lo is None else np.asarray(q_lo, dtype=float)
        hi = np.inf if q_hi is None else np.asarray(q_hi, dtype=float)
        q = np.clip(q, lo, hi)

    if K_p is None:
        K_p = np.diag([5.0, 5.0, 5.0, 7.0, 7.0, 7.0])

    best_q = q.copy()
    best_err = np.inf
    history = {
        "error_norm": [],
        "condition": [],
    }

    for _ in range(max_iter):
        T_cur = body_poe_fk(q, B_list, M)
        e_b = compute_body_error(T_cur, T_des)
        err_norm = float(np.linalg.norm(e_b))
        Jb = body_jacobian(q, B_list)
        cond = float(np.linalg.cond(Jb))

        history["error_norm"].append(err_norm)
        history["condition"].append(cond)

        if err_norm < best_err:
            best_err = err_norm
            best_q = q.copy()
        if err_norm < tol:
            history["best_error_norm"] = best_err
            return q, True, history

        step_damping = damping if np.isfinite(cond) and cond < 1e4 else max(damping, 0.12)
        q, _ = clik_step(
            q,
            T_des,
            B_list,
            M,
            K_p,
            dt,
            damping=step_damping,
            q_lo=q_lo,
            q_hi=q_hi,
            nullspace_gain=nullspace_gain,
            return_info=True,
        )

    history["best_error_norm"] = best_err
    return best_q, False, history

