"""look_at_ik.py - Single-pose look-at IK solvers against the MuJoCo model.

Solves TCP position plus a configurable site-axis look-at constraint,
either by damped-least-squares Jacobian iteration (`solve_look_at_ik`) or
bounded SciPy least-squares (`solve_look_at_ik_optimized`).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - SciPy is available in the target env.
    least_squares = None

from .arm_model import joint_limits, joint_qpos_indices


def _joint_center_terms(q, limits):
    q_mid = 0.5 * (limits[:, 0] + limits[:, 1])
    q_half_range = 0.5 * (limits[:, 1] - limits[:, 0])
    q_half_range = np.maximum(q_half_range, 1e-6)
    normalized = (q - q_mid) / q_half_range
    center_velocity = -(normalized / q_half_range)
    limit_cost = float(np.mean(normalized**2))
    return center_velocity, limit_cost


def _normalized_distance(q, q_ref, limits):
    if q_ref is None:
        return 0.0
    q_range = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)
    return float(np.linalg.norm((q - q_ref) / q_range))


def _solution_score(q, pose_score, q_ref, limits, continuity_weight, limit_weight):
    _, limit_cost = _joint_center_terms(q, limits)
    continuity_cost = _normalized_distance(q, q_ref, limits)
    return pose_score + continuity_weight * continuity_cost + limit_weight * limit_cost


def _look_at_errors(
    model,
    data,
    mujoco_module,
    qidx,
    q,
    site_name,
    target_pos,
    look_target,
    axis_col,
    axis_sign,
):
    data.qpos[qidx] = q
    mujoco_module.mj_forward(model, data)
    sid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, site_name)
    pos = data.site_xpos[sid].copy()
    R = data.site_xmat[sid].reshape(3, 3).copy()
    axis_cur = axis_sign * R[:, axis_col]
    axis_des = look_target - pos
    axis_des /= np.linalg.norm(axis_des) + 1e-12
    pos_err = float(np.linalg.norm(target_pos - pos))
    look_err = float(np.linalg.norm(axis_cur - axis_des))
    return pos_err, look_err


def _look_error_degrees(look_err):
    angle = 2.0 * np.arcsin(np.clip(0.5 * float(look_err), 0.0, 1.0))
    return float(np.rad2deg(angle))


def _capture_gate(pos_err, look_err, max_pos_err, max_look_deg):
    look_deg = _look_error_degrees(look_err)
    capture_valid = (
        float(pos_err) <= float(max_pos_err)
        and look_deg <= float(max_look_deg)
    )
    return capture_valid, look_deg


def _capture_gate_cost(pos_err, look_err, max_pos_err, max_look_deg, invalid_penalty):
    capture_valid, look_deg = _capture_gate(pos_err, look_err, max_pos_err, max_look_deg)
    if capture_valid:
        return 0.0, True, look_deg

    pos_over = max(float(pos_err) - float(max_pos_err), 0.0) / max(float(max_pos_err), 1e-9)
    look_over = max(look_deg - float(max_look_deg), 0.0) / max(float(max_look_deg), 1e-9)
    return float(invalid_penalty) * (1.0 + pos_over**2 + look_over**2), False, look_deg


def _rotation_vector_error(R_current, R_target):
    R_err = R_target @ R_current.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ],
        dtype=float,
    )
    axis /= 2.0 * np.sin(angle) + 1e-12
    return angle * axis


def _site_rotation_error(model, data, mujoco_module, qidx, q, site_name, target_R):
    if target_R is None:
        return 0.0
    data.qpos[qidx] = q
    mujoco_module.mj_forward(model, data)
    sid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, site_name)
    R = data.site_xmat[sid].reshape(3, 3).copy()
    return float(np.linalg.norm(_rotation_vector_error(R, target_R)))


def solve_look_at_ik(
    model,
    data,
    mujoco_module,
    target_pos,
    look_target,
    q_init=None,
    site_name="ee_site",
    axis_col=2,
    axis_sign=-1.0,
    max_iter=900,
    tol_pos=1e-4,
    tol_look=2e-3,
    damping=0.035,
    step=0.35,
    q_ref=None,
    nullspace_center_gain=0.0,
    nullspace_reference_gain=0.0,
    continuity_weight=0.0,
    limit_weight=0.0,
    max_delta=np.inf,
    target_R=None,
    rotation_weight=0.0,
    tol_rot=3e-2,
):
    """Solve TCP position plus a configurable site-axis look-at constraint."""
    qidx = joint_qpos_indices(model, mujoco_module)
    limits = joint_limits(model, mujoco_module)
    sid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site '{site_name}' not found")

    if q_init is None:
        q = data.qpos[qidx].copy()
    else:
        q = np.asarray(q_init, dtype=float).copy()
    q = np.clip(q, limits[:, 0], limits[:, 1])
    if q_ref is not None:
        q_ref = np.clip(np.asarray(q_ref, dtype=float).copy(), limits[:, 0], limits[:, 1])
    target_R = None if target_R is None else np.asarray(target_R, dtype=float).reshape(3, 3)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    best_q = q.copy()
    best_pose_err = np.inf
    best_score = np.inf
    best_ok = False

    for _ in range(max_iter):
        data.qpos[qidx] = q
        mujoco_module.mj_forward(model, data)

        pos = data.site_xpos[sid].copy()
        R = data.site_xmat[sid].reshape(3, 3).copy()
        z_cur = axis_sign * R[:, axis_col]
        z_des = look_target - pos
        z_des /= np.linalg.norm(z_des) + 1e-12

        pos_err = target_pos - pos
        look_err = z_des - z_cur
        if target_R is not None and rotation_weight > 0.0:
            rot_err = _rotation_vector_error(R, target_R)
        else:
            rot_err = np.zeros(3)
        epos = float(np.linalg.norm(pos_err))
        elook = float(np.linalg.norm(look_err))
        erot = float(np.linalg.norm(rot_err))
        pose_score = epos + 0.20 * elook
        score = _solution_score(
            q,
            pose_score,
            q_ref,
            limits,
            continuity_weight,
            limit_weight,
        )
        if score < best_score:
            best_score = score
            best_pose_err = epos
            best_q = q.copy()
            best_ok = epos < tol_pos and elook < tol_look and (
                target_R is None or rotation_weight <= 0.0 or erot < tol_rot
            )
        if epos < tol_pos and elook < tol_look and (
            target_R is None or rotation_weight <= 0.0 or erot < tol_rot
        ):
            return q, True, epos

        mujoco_module.mj_jacSite(model, data, jacp, jacr, sid)
        Jp = jacp[:, qidx]
        Jr = jacr[:, qidx]
        task_rows = [Jp, Jr]
        task_err = [pos_err, 1.25 * np.cross(z_cur, z_des)]
        if target_R is not None and rotation_weight > 0.0:
            task_rows.append(Jr)
            task_err.append(float(rotation_weight) * rot_err)
        J = np.vstack(task_rows)
        err = np.concatenate(task_err)
        A = J @ J.T + damping * damping * np.eye(J.shape[0])
        J_pinv = J.T @ np.linalg.solve(A, np.eye(J.shape[0]))
        dq_task = J_pinv @ err

        center_velocity, _ = _joint_center_terms(q, limits)
        qdot0 = nullspace_center_gain * center_velocity
        if q_ref is not None:
            qdot0 += nullspace_reference_gain * (q_ref - q)

        null_projector = np.eye(len(qidx)) - J_pinv @ J
        dq = dq_task + null_projector @ qdot0
        delta = step * dq
        delta = np.clip(delta, -max_delta, max_delta)
        q = np.clip(q + delta, limits[:, 0], limits[:, 1])

    data.qpos[qidx] = best_q
    mujoco_module.mj_forward(model, data)
    return best_q, best_ok, best_pose_err


def solve_look_at_ik_optimized(
    model,
    data,
    mujoco_module,
    target_pos,
    look_target,
    q_init,
    q_ref,
    site_name="ee_site",
    axis_col=2,
    axis_sign=-1.0,
    tol_pos=5e-4,
    tol_look=2e-3,
    pos_weight=1200.0,
    look_weight=40.0,
    continuity_weight=0.08,
    center_weight=0.035,
    max_nfev=260,
    target_R=None,
    rotation_weight=0.0,
    tol_rot=3e-2,
):
    """Bounded local least-squares IK with continuity and joint-center soft costs."""
    if least_squares is None:
        return solve_look_at_ik(
            model,
            data,
            mujoco_module,
            target_pos,
            look_target,
            q_init=q_init,
            q_ref=q_ref,
            site_name=site_name,
            axis_col=axis_col,
            axis_sign=axis_sign,
            target_R=target_R,
            rotation_weight=rotation_weight,
            tol_rot=tol_rot,
        )

    qidx = joint_qpos_indices(model, mujoco_module)
    limits = joint_limits(model, mujoco_module)
    sid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site '{site_name}' not found")

    q_init = np.clip(np.asarray(q_init, dtype=float), limits[:, 0], limits[:, 1])
    q_ref = np.clip(np.asarray(q_ref, dtype=float), limits[:, 0], limits[:, 1])
    target_R = None if target_R is None else np.asarray(target_R, dtype=float).reshape(3, 3)
    q_mid = 0.5 * (limits[:, 0] + limits[:, 1])
    q_half_range = np.maximum(0.5 * (limits[:, 1] - limits[:, 0]), 1e-6)
    q_range = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)

    def residual(q):
        data.qpos[qidx] = q
        mujoco_module.mj_forward(model, data)
        pos = data.site_xpos[sid].copy()
        R = data.site_xmat[sid].reshape(3, 3).copy()
        z_cur = axis_sign * R[:, axis_col]
        z_des = look_target - pos
        z_des /= np.linalg.norm(z_des) + 1e-12

        pos_res = pos_weight * (pos - target_pos)
        look_res = look_weight * (z_cur - z_des)
        if target_R is not None and rotation_weight > 0.0:
            rot_res = float(rotation_weight) * _rotation_vector_error(R, target_R)
        else:
            rot_res = np.empty(0, dtype=float)
        continuity_res = continuity_weight * ((q - q_ref) / q_range)
        center_res = center_weight * ((q - q_mid) / q_half_range)
        return np.concatenate([pos_res, look_res, rot_res, continuity_res, center_res])

    result = least_squares(
        residual,
        q_init,
        bounds=(limits[:, 0], limits[:, 1]),
        max_nfev=max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    q = np.clip(result.x, limits[:, 0], limits[:, 1])

    data.qpos[qidx] = q
    mujoco_module.mj_forward(model, data)
    pos = data.site_xpos[sid].copy()
    R = data.site_xmat[sid].reshape(3, 3).copy()
    z_cur = axis_sign * R[:, axis_col]
    z_des = look_target - pos
    z_des /= np.linalg.norm(z_des) + 1e-12
    epos = float(np.linalg.norm(target_pos - pos))
    elook = float(np.linalg.norm(z_cur - z_des))
    erot = 0.0 if target_R is None or rotation_weight <= 0.0 else float(
        np.linalg.norm(_rotation_vector_error(R, target_R))
    )
    ok = epos < tol_pos and elook < tol_look and (
        target_R is None or rotation_weight <= 0.0 or erot < tol_rot
    )
    return q, ok, epos
