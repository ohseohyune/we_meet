"""
ik_solver.py — Numerical IK for 7-DOF robot defined by standard DH parameters.

Standard DH convention per joint i:  [a_i, alpha_i, d_i, theta_offset_i]
Transform: T_i = Rz(q_i + offset_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i)

Assumptions
-----------
- Standard DH convention (not modified DH).
- Joint order 1..7, all revolute, rotating about their local z-axis.
- Joint limits: ±2.897 rad for all joints (Panda-like default; adjust per real spec).
- DLS damping lambda=0.05 for singularity robustness.
- IK convergence threshold: 1e-4 m position, 1e-3 rad orientation.
- Maximum iterations: 200 per solve call.
"""

import numpy as np
from typing import Optional, Tuple

# ── DH parameters [a, alpha, d, theta_offset] ──────────────────────────────
DH_PARAMS = np.array([
    [ 0.132184,  3.146991, -0.061099, -1.568837],  # J1
    [ 0.091502,  0.002344,  0.345932, -0.004401],  # J2
    [ 0.005000, -1.577745,  0.103901,  1.574759],  # J3
    [ 0.344628, -0.009289, -0.000588, -1.581564],  # J4
    [-0.024818, -3.104470,  0.074472, -1.553936],  # J5
    [ 0.042524,  1.581094,  0.070882, -1.520796],  # J6
    [-0.244500, -1.570796,  0.000000,  0.000000],  # J7
])

NDOF = 7
JOINT_LIMITS = np.array([[-2.8973, 2.8973]] * 7)  # symmetric ±2.897 rad

# Camera offset in TCP frame (meters).  z-forward, so camera looks along -z of flange.
# Assumption: camera mounted rigidly at TCP, optical axis along TCP +z.
CAMERA_OFFSET_TCP = np.eye(4)


# ── Basic rotation helpers ──────────────────────────────────────────────────

def rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def dh_transform(q: float, a: float, alpha: float, d: float, offset: float) -> np.ndarray:
    """4×4 homogeneous DH transform for one joint."""
    theta = q + offset
    T = np.eye(4)
    Rz_ = rz(theta)
    Rx_ = rx(alpha)
    # T = Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha)
    R = Rz_ @ Rx_
    t = Rz_ @ np.array([a, 0.0, d])
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# ── Forward Kinematics ──────────────────────────────────────────────────────

def fk_all_frames(q: np.ndarray) -> list:
    """Return list of 4×4 transforms T_0_i for i in 0..7 (base=identity, then each joint frame)."""
    frames = [np.eye(4)]
    T = np.eye(4)
    for i in range(NDOF):
        a, alpha, d, offset = DH_PARAMS[i]
        T = T @ dh_transform(q[i], a, alpha, d, offset)
        frames.append(T.copy())
    return frames


def fk(q: np.ndarray) -> np.ndarray:
    """End-effector 4×4 transform in base frame."""
    return fk_all_frames(q)[-1]


# ── Geometric Jacobian ──────────────────────────────────────────────────────

def jacobian(q: np.ndarray) -> np.ndarray:
    """6×7 geometric Jacobian (linear velocity on top, angular on bottom)."""
    frames = fk_all_frames(q)
    p_ee = frames[-1][:3, 3]
    J = np.zeros((6, NDOF))
    for i in range(NDOF):
        z_i = frames[i][:3, 2]          # z-axis of frame i
        p_i = frames[i][:3, 3]          # origin of frame i
        J[:3, i] = np.cross(z_i, p_ee - p_i)   # linear part
        J[3:, i] = z_i                          # angular part
    return J


# ── Pose error ──────────────────────────────────────────────────────────────

def rotation_error(R_current: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
    """Axis-angle error from R_current to R_desired (3-vector, in base frame)."""
    R_err = R_desired @ R_current.T
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace)
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def pose_error(T_current: np.ndarray, T_desired: np.ndarray) -> np.ndarray:
    """6-vector pose error [pos_err (3), rot_err (3)]."""
    pos_err = T_desired[:3, 3] - T_current[:3, 3]
    rot_err = rotation_error(T_current[:3, :3], T_desired[:3, :3])
    return np.concatenate([pos_err, rot_err])


# ── IK Solver ───────────────────────────────────────────────────────────────

def ik_look_at(
    target_pos: np.ndarray,
    look_target: np.ndarray,
    q_init: Optional[np.ndarray] = None,
    max_iter: int = 400,
    tol_pos: float = 1e-4,
    tol_look: float = 1e-3,
    lam: float = 0.05,
    step: float = 0.5,
) -> Tuple[np.ndarray, bool, float]:
    """
    Inspection-specific IK: position + look-at constraint.

    Phase 1: position-only DLS (3 constraints, robust convergence).
    Phase 2: position + look-at direction (cross-product error for z-axis alignment).
             Roll around camera z-axis remains free — matches inspection requirements.

    Parameters
    ----------
    target_pos  : desired TCP position [x, y, z]
    look_target : point the camera z-axis should point toward (flange center)
    q_init      : initial joint angles
    """
    if q_init is None:
        q_init = np.zeros(NDOF)
    q = q_init.copy()

    # Phase 1: position-only
    iter1 = int(max_iter * 0.5)
    for _ in range(iter1):
        T_cur = fk(q)
        pos_err = target_pos - T_cur[:3, 3]
        if np.linalg.norm(pos_err) < tol_pos:
            break
        J3 = jacobian(q)[:3, :]
        JJT = J3 @ J3.T
        dq = J3.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), pos_err)
        q = np.clip(q + step * dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    # Phase 2: position + look-at
    iter2 = max_iter - iter1
    W = np.diag([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
    for _ in range(iter2):
        T_cur = fk(q)
        pos_err = target_pos - T_cur[:3, 3]
        z_cur = T_cur[:3, 2]
        dz = look_target - T_cur[:3, 3]
        z_des = dz / (np.linalg.norm(dz) + 1e-12)
        look_err = np.cross(z_cur, z_des)  # zero when z_cur aligns with z_des
        err = np.concatenate([pos_err, look_err])
        e_pos  = float(np.linalg.norm(pos_err))
        e_look = float(np.linalg.norm(look_err))
        if e_pos < tol_pos and e_look < tol_look:
            return q, True, e_pos
        J = jacobian(q)
        JW = J.T @ W
        JJT = JW @ J
        dq = np.linalg.solve(JJT + lam**2 * np.eye(NDOF), JW @ err)
        q = np.clip(q + step * dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    T_cur = fk(q)
    e_pos = float(np.linalg.norm(target_pos - T_cur[:3, 3]))
    z_cur = T_cur[:3, 2]
    dz = look_target - T_cur[:3, 3]
    z_des = dz / (np.linalg.norm(dz) + 1e-12)
    e_look = float(np.linalg.norm(np.cross(z_cur, z_des)))
    success = e_pos < tol_pos and e_look < tol_look
    return q, success, e_pos


def ik(
    T_desired: np.ndarray,
    q_init: Optional[np.ndarray] = None,
    max_iter: int = 400,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-2,
    lam: float = 0.05,
    step: float = 0.5,
    w_pos: float = 1.0,
    w_rot: float = 0.1,
) -> Tuple[np.ndarray, bool, float]:
    """
    Two-phase Damped Least Squares IK.

    Phase 1 (position-only): drives TCP to target position quickly.
    Phase 2 (full 6-DOF):    refines orientation using weighted error.

    Parameters
    ----------
    T_desired : 4×4 desired end-effector pose in base frame
    q_init    : initial joint config (radians). Defaults to zeros.
    max_iter  : total iterations (split 60/40 between phase 1 and 2)
    tol_pos   : positional convergence threshold (m)
    tol_rot   : orientation convergence threshold (rad)
    lam       : DLS damping factor
    step      : gradient step scale
    w_pos     : position error weight
    w_rot     : orientation error weight (lower = softer orientation constraint)

    Returns
    -------
    q_sol  : 7-vector joint angles (radians)
    success: bool
    err    : final position error norm
    """
    if q_init is None:
        q_init = np.zeros(NDOF)
    q = q_init.copy()

    W = np.diag([w_pos, w_pos, w_pos, w_rot, w_rot, w_rot])

    # ── Phase 1: position-only IK ────────────────────────────────────────
    iter1 = int(max_iter * 0.6)
    for _ in range(iter1):
        T_cur = fk(q)
        pos_err = T_desired[:3, 3] - T_cur[:3, 3]
        e_pos = np.linalg.norm(pos_err)
        if e_pos < tol_pos:
            break
        J = jacobian(q)
        J3 = J[:3, :]
        JJT = J3 @ J3.T
        dq = J3.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), pos_err)
        q = np.clip(q + step * dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    # ── Phase 2: full weighted 6-DOF refinement ──────────────────────────
    iter2 = max_iter - iter1
    for _ in range(iter2):
        T_cur = fk(q)
        err = pose_error(T_cur, T_desired)
        e_pos = np.linalg.norm(err[:3])
        e_rot = np.linalg.norm(err[3:])
        if e_pos < tol_pos and e_rot < tol_rot:
            return q, True, float(e_pos)
        werr = W @ err
        J = jacobian(q)
        JW = J.T @ W
        JJT = JW @ J
        dq = np.linalg.solve(JJT + lam**2 * np.eye(NDOF), JW @ err)
        q = np.clip(q + step * dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    T_cur = fk(q)
    e_pos = float(np.linalg.norm(T_desired[:3, 3] - T_cur[:3, 3]))
    e_rot = float(np.linalg.norm(rotation_error(T_cur[:3, :3], T_desired[:3, :3])))
    success = e_pos < tol_pos and e_rot < tol_rot
    return q, success, e_pos


def ik_trajectory_multiseed(
    poses: list,
    q_start: Optional[np.ndarray] = None,
    verbose: bool = True,
    look_target: Optional[np.ndarray] = None,
    n_retries: int = 8,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, list]:
    """
    IK trajectory solver with random-restart fallback for failed waypoints.

    For each waypoint:
      1. Try with sequential seed from previous waypoint.
      2. If fails, retry up to n_retries times with perturbed / random seeds.
      3. Use best result (lowest position error) among all trials.
    """
    rng = np.random.default_rng(rng_seed)
    N = len(poses)
    Q = np.zeros((N, NDOF))
    flags = []
    q = np.zeros(NDOF) if q_start is None else q_start.copy()
    q_prev_ok = q.copy()

    for i, T_des in enumerate(poses):
        target_pos = T_des[:3, 3]
        best_q, best_ok, best_err = q.copy(), False, 1e9

        # Primary: sequential seed
        if look_target is not None:
            q_sol, ok, err = ik_look_at(target_pos, look_target, q_init=q)
        else:
            q_sol, ok, err = ik(T_des, q_init=q)
        if err < best_err:
            best_q, best_ok, best_err = q_sol.copy(), ok, err

        # Retries if failed
        if not best_ok:
            for trial in range(n_retries):
                if trial % 2 == 0:
                    seed = q_prev_ok + rng.uniform(-0.5, 0.5, NDOF)
                else:
                    seed = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
                seed = np.clip(seed, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
                if look_target is not None:
                    q_sol, ok, err = ik_look_at(target_pos, look_target, q_init=seed)
                else:
                    q_sol, ok, err = ik(T_des, q_init=seed)
                if err < best_err:
                    best_q, best_ok, best_err = q_sol.copy(), ok, err
                if best_ok:
                    break

        Q[i] = best_q
        flags.append(best_ok)
        if best_ok:
            q = best_q.copy()
            q_prev_ok = best_q.copy()
        else:
            q = best_q.copy()  # use best guess anyway to seed next

        if verbose:
            status = "OK  " if best_ok else "FAIL"
            print(f"  WP {i+1:3d}/{N}: [{status}] pos_err={best_err*1000:.2f}mm")

    return Q, flags


def ik_trajectory(
    poses: list,
    q_start: Optional[np.ndarray] = None,
    verbose: bool = True,
    look_target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, list]:
    """
    Solve IK for a sequence of poses, seeding each from the previous solution.

    If `look_target` is provided (e.g., flange center), uses ik_look_at()
    instead of the full 6-DOF ik() — more robust for inspection orbits.

    Parameters
    ----------
    poses       : list of 4×4 ndarray target poses
    q_start     : initial joint config
    look_target : 3D point camera should look toward (enables look-at IK)

    Returns
    -------
    Q     : (N, 7) joint trajectory
    flags : list of bool success flags
    """
    N = len(poses)
    Q = np.zeros((N, NDOF))
    flags = []
    q = np.zeros(NDOF) if q_start is None else q_start.copy()

    for i, T_des in enumerate(poses):
        if look_target is not None:
            q_sol, ok, err = ik_look_at(T_des[:3, 3], look_target, q_init=q)
        else:
            q_sol, ok, err = ik(T_des, q_init=q)
        Q[i] = q_sol
        flags.append(ok)
        q = q_sol
        if verbose:
            status = "OK" if ok else "FAIL"
            print(f"  Waypoint {i+1:3d}/{N}: [{status}] pos_err={err*1000:.2f}mm")

    return Q, flags


# ── Quaternion helpers ───────────────────────────────────────────────────────

def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z]."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


if __name__ == "__main__":
    # Smoke test: FK at zero angles, then IK back to that pose
    q0 = np.zeros(NDOF)
    T0 = fk(q0)
    print("FK at q=0:")
    print(np.round(T0, 4))

    q_sol, ok, err = ik(T0, q_init=q0 + 0.1)
    print(f"\nIK recovery: success={ok}, error={err:.6f}")
    print("q_sol:", np.round(q_sol, 4))
    print("q_ref:", np.round(q0, 4))
