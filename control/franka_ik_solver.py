"""
franka_ik_solver.py - MuJoCo Jacobian IK for the real Franka Panda scene.

This intentionally does not use the supplied DH parameters.  It solves IK
directly against the loaded MuJoCo Franka model and the `tcp` site.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - SciPy is available in the target env.
    least_squares = None


FRANKA_JOINT_NAMES = [f"panda0_joint{i}" for i in range(1, 8)]


def joint_qpos_indices(model, mujoco_module):
    ids = [mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, n) for n in FRANKA_JOINT_NAMES]
    if any(i < 0 for i in ids):
        missing = [n for n, i in zip(FRANKA_JOINT_NAMES, ids) if i < 0]
        raise ValueError(f"Missing Franka joints in model: {missing}")
    return np.array([model.jnt_qposadr[i] for i in ids], dtype=int)


def joint_limits(model, mujoco_module):
    ids = [mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, n) for n in FRANKA_JOINT_NAMES]
    return np.array([model.jnt_range[i] for i in ids], dtype=float)


def set_arm_qpos(model, data, mujoco_module, q):
    idx = joint_qpos_indices(model, mujoco_module)
    data.qpos[idx] = q
    mujoco_module.mj_forward(model, data)


def get_arm_qpos(model, data, mujoco_module):
    idx = joint_qpos_indices(model, mujoco_module)
    return data.qpos[idx].copy()


def site_pose(model, data, mujoco_module, site_name="tcp"):
    sid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site '{site_name}' not found")
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[sid]
    T[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    return T


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


def solve_look_at_ik(
    model,
    data,
    mujoco_module,
    target_pos,
    look_target,
    q_init=None,
    site_name="tcp",
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
):
    """Solve TCP position + TCP +z look-at alignment using MuJoCo site Jacobian."""
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
        look_err = np.cross(z_cur, z_des)
        epos = float(np.linalg.norm(pos_err))
        elook = float(np.linalg.norm(look_err))
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
            best_ok = epos < tol_pos and elook < tol_look
        if epos < tol_pos and elook < tol_look:
            return q, True, epos

        mujoco_module.mj_jacSite(model, data, jacp, jacr, sid)
        Jp = jacp[:, qidx]
        Jr = jacr[:, qidx]
        J = np.vstack([Jp, Jr])
        err = np.concatenate([pos_err, 1.25 * look_err])
        A = J @ J.T + damping * damping * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
        dq_task = J_pinv @ err

        center_velocity, _ = _joint_center_terms(q, limits)
        qdot0 = nullspace_center_gain * center_velocity
        if q_ref is not None:
            qdot0 += nullspace_reference_gain * (q_ref - q)

        null_projector = np.eye(7) - J_pinv @ J
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
    site_name="tcp",
    axis_col=2,
    axis_sign=-1.0,
    tol_pos=1e-4,
    tol_look=2e-3,
    pos_weight=1200.0,
    look_weight=8.0,
    continuity_weight=0.18,
    center_weight=0.035,
    max_nfev=140,
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
        )

    qidx = joint_qpos_indices(model, mujoco_module)
    limits = joint_limits(model, mujoco_module)
    sid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site '{site_name}' not found")

    q_init = np.clip(np.asarray(q_init, dtype=float), limits[:, 0], limits[:, 1])
    q_ref = np.clip(np.asarray(q_ref, dtype=float), limits[:, 0], limits[:, 1])
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
        look_res = look_weight * np.cross(z_cur, z_des)
        continuity_res = continuity_weight * ((q - q_ref) / q_range)
        center_res = center_weight * ((q - q_mid) / q_half_range)
        return np.concatenate([pos_res, look_res, continuity_res, center_res])

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
    elook = float(np.linalg.norm(np.cross(z_cur, z_des)))
    ok = epos < tol_pos and elook < tol_look
    return q, ok, epos


def solve_pose_ik_optimized(
    model,
    data,
    mujoco_module,
    target_pose,
    q_init,
    q_ref,
    site_name="tcp",
    tol_pos=1e-4,
    tol_rot=1e-3,
    pos_weight=1200.0,
    rot_weight=8.0,
    continuity_weight=0.18,
    center_weight=0.035,
    max_nfev=180,
):
    """Bounded local least-squares IK for a full 6D site pose."""
    if least_squares is None:
        raise ImportError("scipy is required for full-pose MuJoCo IK.")

    qidx = joint_qpos_indices(model, mujoco_module)
    limits = joint_limits(model, mujoco_module)
    sid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site '{site_name}' not found")

    target_pose = np.asarray(target_pose, dtype=float)
    target_pos = target_pose[:3, 3].copy()
    target_R = target_pose[:3, :3].copy()
    q_init = np.clip(np.asarray(q_init, dtype=float), limits[:, 0], limits[:, 1])
    q_ref = np.clip(np.asarray(q_ref, dtype=float), limits[:, 0], limits[:, 1])
    q_mid = 0.5 * (limits[:, 0] + limits[:, 1])
    q_half_range = np.maximum(0.5 * (limits[:, 1] - limits[:, 0]), 1e-6)
    q_range = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)

    def residual(q):
        data.qpos[qidx] = q
        mujoco_module.mj_forward(model, data)
        pos = data.site_xpos[sid].copy()
        R = data.site_xmat[sid].reshape(3, 3).copy()

        pos_res = pos_weight * (pos - target_pos)
        rot_res = rot_weight * _rotation_vector_error(R, target_R)
        continuity_res = continuity_weight * ((q - q_ref) / q_range)
        center_res = center_weight * ((q - q_mid) / q_half_range)
        return np.concatenate([pos_res, rot_res, continuity_res, center_res])

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
    epos = float(np.linalg.norm(target_pos - pos))
    erot = float(np.linalg.norm(_rotation_vector_error(R, target_R)))
    ok = epos < tol_pos and erot < tol_rot
    return q, ok, epos + 0.20 * erot


def solve_trajectory(
    model,
    data,
    mujoco_module,
    tcp_poses,
    look_target,
    q_start=None,
    retries=12,
    rng_seed=7,
    verbose=True,
    axis_col=2,
    axis_sign=-1.0,
    continuity_weight=0.18,
    limit_weight=0.035,
    method="least_squares",
):
    rng = np.random.default_rng(rng_seed)
    limits = joint_limits(model, mujoco_module)
    q = np.zeros(7) if q_start is None else np.asarray(q_start, dtype=float).copy()
    Q = np.zeros((len(tcp_poses), 7))
    flags = []
    look_target_arr = np.asarray(look_target, dtype=float)
    if look_target_arr.shape == (3,):
        look_targets = np.tile(look_target_arr.reshape(1, 3), (len(tcp_poses), 1))
    elif look_target_arr.shape == (len(tcp_poses), 3):
        look_targets = look_target_arr
    else:
        raise ValueError(
            "look_target must be either shape (3,) or one target per waypoint "
            f"with shape ({len(tcp_poses)}, 3); got {look_target_arr.shape}."
        )

    for i, (T, target_i) in enumerate(zip(tcp_poses, look_targets)):
        target_pos = T[:3, 3]
        solver = solve_look_at_ik_optimized if method == "least_squares" else solve_look_at_ik
        best_q, best_ok, best_err = solver(
            model, data, mujoco_module, target_pos, target_i,
            q_init=q, q_ref=q, axis_col=axis_col, axis_sign=axis_sign,
            continuity_weight=continuity_weight, center_weight=limit_weight,
        ) if solver is solve_look_at_ik_optimized else solver(
            model, data, mujoco_module, target_pos, target_i,
            q_init=q, q_ref=q, axis_col=axis_col, axis_sign=axis_sign,
            continuity_weight=continuity_weight, limit_weight=limit_weight,
        )
        best_score = _solution_score(best_q, best_err, q, limits, continuity_weight, limit_weight)

        for trial in range(retries):
            if best_ok:
                break
            if trial % 2 == 0:
                seed = q + rng.uniform(-0.35, 0.35, 7)
            else:
                seed = rng.uniform(limits[:, 0], limits[:, 1])
            seed = np.clip(seed, limits[:, 0], limits[:, 1])
            q_try, ok, err = solver(
                model, data, mujoco_module, target_pos, target_i,
                q_init=seed, q_ref=q, axis_col=axis_col, axis_sign=axis_sign,
                continuity_weight=continuity_weight, center_weight=limit_weight,
            ) if solver is solve_look_at_ik_optimized else solver(
                model, data, mujoco_module, target_pos, target_i,
                q_init=seed, q_ref=q, axis_col=axis_col, axis_sign=axis_sign,
                continuity_weight=continuity_weight, limit_weight=limit_weight,
            )
            score = _solution_score(q_try, err, q, limits, continuity_weight, limit_weight)
            if score < best_score:
                best_q, best_ok, best_err, best_score = q_try, ok, err, score

        Q[i] = best_q
        flags.append(best_ok)
        q = best_q.copy()

        if verbose:
            status = "OK  " if best_ok else "FAIL"
            print(f"  WP {i+1:3d}/{len(tcp_poses)}: [{status}] pos_err={best_err*1000:.2f}mm")

    return Q, flags


def solve_pose_trajectory(
    model,
    data,
    mujoco_module,
    target_poses,
    q_start=None,
    retries=12,
    rng_seed=7,
    verbose=True,
    continuity_weight=0.18,
    limit_weight=0.035,
    site_name="tcp",
):
    rng = np.random.default_rng(rng_seed)
    limits = joint_limits(model, mujoco_module)
    q = np.zeros(7) if q_start is None else np.asarray(q_start, dtype=float).copy()
    Q = np.zeros((len(target_poses), 7))
    flags = []

    for i, T in enumerate(target_poses):
        best_q, best_ok, best_err = solve_pose_ik_optimized(
            model,
            data,
            mujoco_module,
            T,
            q_init=q,
            q_ref=q,
            continuity_weight=continuity_weight,
            center_weight=limit_weight,
            site_name=site_name,
        )
        best_score = _solution_score(best_q, best_err, q, limits, continuity_weight, limit_weight)

        for trial in range(retries):
            if best_ok:
                break
            seed = q + rng.uniform(-0.35, 0.35, 7) if trial % 2 == 0 else rng.uniform(limits[:, 0], limits[:, 1])
            seed = np.clip(seed, limits[:, 0], limits[:, 1])
            q_try, ok, err = solve_pose_ik_optimized(
                model,
                data,
                mujoco_module,
                T,
                q_init=seed,
                q_ref=q,
                continuity_weight=continuity_weight,
                center_weight=limit_weight,
                site_name=site_name,
            )
            score = _solution_score(q_try, err, q, limits, continuity_weight, limit_weight)
            if score < best_score:
                best_q, best_ok, best_err, best_score = q_try, ok, err, score

        Q[i] = best_q
        flags.append(best_ok)
        q = best_q.copy()

        if verbose:
            status = "OK  " if best_ok else "FAIL"
            print(f"  WP {i+1:3d}/{len(target_poses)}: [{status}] pose_err={best_err:.5f}")

    return Q, flags
