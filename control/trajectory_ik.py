"""trajectory_ik.py - Trajectory-level IK solving, evaluation, and retiming.

Generates joint trajectories for waypoint sequences with look-at
constraints, including DP-based path selection over per-waypoint IK
candidates (`solve_trajectory` with `path_planning=True`), plus
motion metrics and time-parameterization utilities.
"""

from __future__ import annotations

import numpy as np

from .arm_model import collision_contacts_for_q, joint_limits, joint_qpos_indices
from .look_at_ik import (
    _capture_gate,
    _capture_gate_cost,
    _joint_center_terms,
    _look_at_errors,
    _normalized_distance,
    _site_rotation_error,
    _solution_score,
    solve_look_at_ik,
    solve_look_at_ik_optimized,
)


def joint_motion_metrics(time_values, Q):
    """궤적과 시간이 배열로 주어졌을 때, 속도, 가속도, 저크의 최댓값을 반환해(유한차분법)"""
    time_values = np.asarray(time_values, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if len(time_values) != len(Q):
        raise ValueError("Q and time_values must have the same length.")
    if len(time_values) < 2:
        # 데이터가 2개 미만이면, 속도와 가속도를 구할 수 없으므로 모두 0으로 반환
        return {
            "max_speed": 0.0,
            "max_accel": 0.0,
            "max_jerk": 0.0,
            "duration": 0.0,
        }

    dt = np.maximum(np.diff(time_values), 1e-9)
    dq = np.diff(Q, axis=0)
    velocity = dq / dt[:, None]
    max_speed = float(np.max(np.abs(velocity))) if len(velocity) else 0.0

    if len(velocity) < 2:
        max_accel = 0.0
        max_jerk = 0.0
    else:
        accel_dt = np.maximum(0.5 * (dt[1:] + dt[:-1]), 1e-9)
        accel = np.diff(velocity, axis=0) / accel_dt[:, None]
        max_accel = float(np.max(np.abs(accel))) if len(accel) else 0.0
        if len(accel) < 2:
            max_jerk = 0.0
        else:
            jerk_dt = np.maximum(0.5 * (accel_dt[1:] + accel_dt[:-1]), 1e-9)
            jerk = np.diff(accel, axis=0) / jerk_dt[:, None]
            max_jerk = float(np.max(np.abs(jerk))) if len(jerk) else 0.0

    return {
        "max_speed": max_speed,
        "max_accel": max_accel,
        "max_jerk": max_jerk,
        "duration": float(time_values[-1] - time_values[0]),
    }


def _positive_limit(value):
    "양수면 통과 아니면 none 반환."
    if value is None:
        return None
    value = float(value)
    return value if value > 0.0 else None


def retime_joint_trajectory(
    time_values,
    Q,
    max_joint_speed=0.85,
    max_joint_accel=None,
    max_joint_jerk=None,
    max_iterations=60,
):
    """Stretch waypoint times to respect joint speed, acceleration, and jerk limits."""
    time_values = np.asarray(time_values, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if len(time_values) < 2:
        return time_values.copy()
    if len(Q) != len(time_values):
        raise ValueError("Q and time_values must have the same length.")

    dt = np.maximum(np.diff(time_values), 1e-6)
    dq = np.diff(Q, axis=0)

    max_joint_speed = _positive_limit(max_joint_speed)
    max_joint_accel = _positive_limit(max_joint_accel)
    max_joint_jerk = _positive_limit(max_joint_jerk)

    if max_joint_speed is not None:
        required_dt = np.max(np.abs(dq), axis=1) / max(max_joint_speed, 1e-9)
        dt = np.maximum(dt, required_dt)

    for _ in range(max(1, int(max_iterations))):
        changed = False
        velocity = dq / dt[:, None]

        if max_joint_accel is not None and len(velocity) >= 2:
            accel_dt = np.maximum(0.5 * (dt[1:] + dt[:-1]), 1e-9)
            accel = np.diff(velocity, axis=0) / accel_dt[:, None]
            accel_peak = np.max(np.abs(accel), axis=1)
            interval_scale = np.ones_like(dt)
            for i, peak in enumerate(accel_peak):
                if peak <= max_joint_accel * 1.001:
                    continue
                scale = min(3.0, max(1.02, np.sqrt(peak / max_joint_accel) * 1.02))
                interval_scale[i:i + 2] = np.maximum(interval_scale[i:i + 2], scale)
            if np.any(interval_scale > 1.0):
                dt *= interval_scale
                changed = True
                velocity = dq / dt[:, None]

        if max_joint_jerk is not None and len(velocity) >= 3:
            accel_dt = np.maximum(0.5 * (dt[1:] + dt[:-1]), 1e-9)
            accel = np.diff(velocity, axis=0) / accel_dt[:, None]
            jerk_dt = np.maximum(0.5 * (accel_dt[1:] + accel_dt[:-1]), 1e-9)
            jerk = np.diff(accel, axis=0) / jerk_dt[:, None]
            jerk_peak = np.max(np.abs(jerk), axis=1)
            interval_scale = np.ones_like(dt)
            for i, peak in enumerate(jerk_peak):
                if peak <= max_joint_jerk * 1.001:
                    continue
                scale = min(3.0, max(1.02, np.cbrt(peak / max_joint_jerk) * 1.02))
                interval_scale[i:i + 3] = np.maximum(interval_scale[i:i + 3], scale)
            if np.any(interval_scale > 1.0):
                dt *= interval_scale
                changed = True

        if not changed:
            break

    new_time = np.empty_like(time_values)
    new_time[0] = time_values[0]
    new_time[1:] = time_values[0] + np.cumsum(dt)
    return new_time


def _joint_step_cost(q, q_ref, max_joint_step):
    if q_ref is None or max_joint_step is None or max_joint_step <= 0.0:
        return 0.0
    delta = np.abs(np.asarray(q, dtype=float) - np.asarray(q_ref, dtype=float))
    excess = np.maximum(delta - float(max_joint_step), 0.0)
    return float(np.mean(excess**2))


def _periodic_joint_mask(limits):
    # The active arm joints are limited hinges.  Treating +/-pi ranges as
    # periodic caused unwrapped solutions outside the physical joint limits.
    return np.zeros(len(limits), dtype=bool)


def _shortest_joint_delta(q_to, q_from, periodic_mask=None):
    delta = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
    if periodic_mask is not None and np.any(periodic_mask):
        delta = delta.copy()
        delta[periodic_mask] = (delta[periodic_mask] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def _unwrap_joint_path(Q, periodic_mask):
    Q = np.asarray(Q, dtype=float)
    if len(Q) < 2 or periodic_mask is None or not np.any(periodic_mask):
        return Q.copy()
    unwrapped = Q.copy()
    for i in range(1, len(Q)):
        unwrapped[i] = unwrapped[i - 1] + _shortest_joint_delta(Q[i], Q[i - 1], periodic_mask)
    return unwrapped


def _posture_bias_cost(model, data, mujoco_module, qidx, q, limits, posture_bias):
    if not posture_bias:
        return 0.0

    data.qpos[qidx] = q
    mujoco_module.mj_forward(model, data)

    cost = 0.0
    q_ref = posture_bias.get("q_ref")
    if q_ref is not None:
        q_ref = np.clip(np.asarray(q_ref, dtype=float), limits[:, 0], limits[:, 1])
        cost += float(posture_bias.get("q_weight", 0.0)) * _normalized_distance(q, q_ref, limits)

    body_names = posture_bias.get("body_names", ())
    if body_names:
        side_axis = int(posture_bias.get("side_axis", 1))
        side_sign = float(posture_bias.get("side_sign", 1.0))
        side_margin = float(posture_bias.get("side_margin", 0.0))
        body_weight = float(posture_bias.get("body_weight", 0.0))
        deficits = []
        for body_name in body_names:
            body_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            side_value = side_sign * float(data.xpos[body_id, side_axis])
            deficits.append(max(side_margin - side_value, 0.0))
        if deficits:
            cost += body_weight * float(np.mean(deficits))

    return cost


def evaluate_look_at_trajectory(
    model,
    data,
    mujoco_module,
    Q,
    tcp_poses,
    look_targets,
    axis_col=2,
    axis_sign=-1.0,
    site_name="ee_site",
    max_pos_err=0.010,
    max_look_deg=5.0,
):
    """Return per-waypoint tracking/look-at metrics and capture-valid flags."""
    qidx = joint_qpos_indices(model, mujoco_module)
    look_targets = np.asarray(look_targets, dtype=float)
    if look_targets.shape == (3,):
        look_targets = np.tile(look_targets.reshape(1, 3), (len(Q), 1))

    pos_errs = []
    look_errs = []
    look_degs = []
    capture_valid = []
    for q, T, target in zip(Q, tcp_poses, look_targets):
        pos_err, look_err = _look_at_errors(
            model,
            data,
            mujoco_module,
            qidx,
            q,
            site_name,
            T[:3, 3],
            target,
            axis_col,
            axis_sign,
        )
        valid, look_deg = _capture_gate(pos_err, look_err, max_pos_err, max_look_deg)
        pos_errs.append(pos_err)
        look_errs.append(look_err)
        look_degs.append(look_deg)
        capture_valid.append(valid)

    return {
        "pos_err": np.asarray(pos_errs, dtype=float),
        "look_err": np.asarray(look_errs, dtype=float),
        "look_deg": np.asarray(look_degs, dtype=float),
        "capture_valid": np.asarray(capture_valid, dtype=bool),
    }


def _joint_transition_cost(
    q_to,
    q_from,
    max_joint_step,
    transition_weight,
    joint_step_weight,
    periodic_mask=None,
):
    if q_from is None:
        return 0.0
    delta = np.abs(_shortest_joint_delta(q_to, q_from, periodic_mask))

    step_scale = float(max_joint_step) if max_joint_step is not None and max_joint_step > 0.0 else 0.55
    step_scale = max(step_scale, 1e-6)
    smooth_cost = float(transition_weight) * float(np.mean((delta / step_scale) ** 2))
    excess = np.maximum(delta - step_scale, 0.0) / step_scale
    excess_cost = float(joint_step_weight) * float(np.mean(excess**2))
    return smooth_cost + excess_cost


def _append_unique_candidate(candidates, candidate, duplicate_tol=0.035):
    q = candidate["q"]
    for idx, existing in enumerate(candidates):
        if np.linalg.norm(existing["q"] - q) < duplicate_tol:
            if candidate["node_cost"] < existing["node_cost"]:
                candidates[idx] = candidate
            return
    candidates.append(candidate)


def _ranked_candidates(candidates, limit):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda item: item["node_cost"])
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return candidates

    selected = candidates[:limit]
    # Keep a little diversity when one IK branch has many tiny local variants.
    for candidate in candidates[limit:]:
        if all(np.linalg.norm(candidate["q"] - other["q"]) > 0.28 for other in selected):
            selected[-1] = candidate
            selected = sorted(selected, key=lambda item: item["node_cost"])
    return selected


def _solve_trajectory_path_dp(
    model,
    data,
    mujoco_module,
    tcp_poses,
    look_targets,
    q_start,
    retries,
    rng,
    verbose,
    axis_col,
    axis_sign,
    continuity_weight,
    limit_weight,
    method,
    posture_bias,
    posture_weights_arr,
    site_name,
    max_joint_step,
    joint_step_weight,
    candidate_limit,
    transition_weight,
    pose_position_scale,
    pose_look_scale,
    pose_rotation_scale,
    target_rotation_weight,
    target_rotation_tol,
    max_capture_look_deg,
    max_capture_pos_err,
    invalid_candidate_penalty,
    collision_penalty,
    collision_margin,
    progress_cb=None,
):
    """Generate per-waypoint IK candidates, then choose the smoothest path.

    progress_cb: optional callable(i, n) invoked after each waypoint's
    candidate generation (the dominant cost) for progress reporting.
    """
    limits = joint_limits(model, mujoco_module)
    qidx = joint_qpos_indices(model, mujoco_module)
    ndof = len(limits)
    periodic_mask = _periodic_joint_mask(limits)
    q_start = np.clip(np.asarray(q_start, dtype=float), limits[:, 0], limits[:, 1])
    q_mid = 0.5 * (limits[:, 0] + limits[:, 1])
    q_zero = np.clip(np.zeros(ndof), limits[:, 0], limits[:, 1])
    solver = solve_look_at_ik_optimized if method == "least_squares" else solve_look_at_ik

    bias_q = posture_bias.get("q_ref") if posture_bias else None
    if bias_q is not None:
        bias_q = np.clip(np.asarray(bias_q, dtype=float), limits[:, 0], limits[:, 1])

    def solve_from_seed(target_pos, target_R, target_i, seed, q_ref, posture_weight):
        seed = np.clip(np.asarray(seed, dtype=float), limits[:, 0], limits[:, 1])
        q_ref = np.clip(np.asarray(q_ref, dtype=float), limits[:, 0], limits[:, 1])
        q_try, ok, _ = solver(
            model,
            data,
            mujoco_module,
            target_pos,
            target_i,
            q_init=seed,
            q_ref=q_ref,
            site_name=site_name,
            axis_col=axis_col,
            axis_sign=axis_sign,
            continuity_weight=continuity_weight,
            center_weight=limit_weight,
            target_R=target_R,
            rotation_weight=target_rotation_weight,
            tol_rot=target_rotation_tol,
        ) if solver is solve_look_at_ik_optimized else solver(
            model,
            data,
            mujoco_module,
            target_pos,
            target_i,
            q_init=seed,
            q_ref=q_ref,
            site_name=site_name,
            axis_col=axis_col,
            axis_sign=axis_sign,
            continuity_weight=continuity_weight,
            limit_weight=limit_weight,
            target_R=target_R,
            rotation_weight=target_rotation_weight,
            tol_rot=target_rotation_tol,
        )
        pos_err, look_err = _look_at_errors(
            model,
            data,
            mujoco_module,
            qidx,
            q_try,
            site_name,
            target_pos,
            target_i,
            axis_col,
            axis_sign,
        )
        rot_err = _site_rotation_error(
            model,
            data,
            mujoco_module,
            qidx,
            q_try,
            site_name,
            target_R,
        )
        collision = collision_contacts_for_q(
            model,
            data,
            mujoco_module,
            q_try,
            qidx=qidx,
            collision_margin=collision_margin,
            max_pairs=3,
        )
        pos_cost = (pos_err / max(float(pose_position_scale), 1e-9)) ** 2
        look_cost = 2.0 * (look_err / max(float(pose_look_scale), 1e-9)) ** 2
        rot_cost = (rot_err / max(float(pose_rotation_scale), 1e-9)) ** 2
        collision_cost = float(collision_penalty) * float(collision["count"])
        node_cost = pos_cost + look_cost + rot_cost + collision_cost
        gate_cost, capture_valid, look_deg = _capture_gate_cost(
            pos_err,
            look_err,
            max_capture_pos_err,
            max_capture_look_deg,
            invalid_candidate_penalty,
        )
        node_cost += gate_cost
        _, limit_cost = _joint_center_terms(q_try, limits)
        node_cost += float(limit_weight) * limit_cost
        if posture_weight > 0.0 and posture_bias:
            node_cost += posture_weight * _posture_bias_cost(
                model,
                data,
                mujoco_module,
                qidx,
                q_try,
                limits,
                posture_bias,
            )

        return {
            "q": q_try,
            "ok": bool(ok),
            "pos_err": pos_err,
            "look_err": look_err,
            "look_deg": look_deg,
            "rot_err": rot_err,
            "collision_count": int(collision["count"]),
            "collision_contacts": collision["contacts"],
            "capture_valid": bool(capture_valid),
            "node_cost": float(node_cost),
        }

    candidate_sets = []
    anchor_q = q_start.copy()
    for i, (T, target_i) in enumerate(zip(tcp_poses, look_targets)):
        target_pos = T[:3, 3]
        target_R = T[:3, :3]
        posture_weight = float(posture_weights_arr[i])
        candidates = []

        deterministic_seeds = [
            (anchor_q, anchor_q),
            (q_start, q_start),
            (q_mid, q_mid),
            (q_zero, q_zero),
        ]
        if bias_q is not None:
            deterministic_seeds.extend([(bias_q, bias_q), (0.5 * (anchor_q + bias_q), anchor_q)])

        for seed, q_ref in deterministic_seeds:
            candidate = solve_from_seed(target_pos, target_R, target_i, seed, q_ref, posture_weight)
            _append_unique_candidate(candidates, candidate)

        for trial in range(retries):
            if bias_q is not None and posture_weight > 0.0 and trial % 4 == 1:
                seed = bias_q + rng.uniform(-0.45, 0.45, ndof)
                q_ref = bias_q
            elif trial % 4 == 0:
                seed = anchor_q + rng.uniform(-0.45, 0.45, ndof)
                q_ref = anchor_q
            elif trial % 4 == 2:
                seed = q_start + rng.uniform(-0.75, 0.75, ndof)
                q_ref = q_start
            else:
                seed = rng.uniform(limits[:, 0], limits[:, 1])
                q_ref = seed
            candidate = solve_from_seed(target_pos, target_R, target_i, seed, q_ref, posture_weight)
            _append_unique_candidate(candidates, candidate)

        ranked = _ranked_candidates(candidates, candidate_limit)
        if not ranked:
            fallback = solve_from_seed(target_pos, target_R, target_i, anchor_q, anchor_q, posture_weight)
            ranked = [fallback]
        candidate_sets.append(ranked)

        anchor_costs = [
            candidate["node_cost"]
            + _joint_transition_cost(
                candidate["q"],
                anchor_q,
                max_joint_step,
                transition_weight,
                joint_step_weight,
                periodic_mask,
            )
            for candidate in ranked
        ]
        anchor_q = ranked[int(np.argmin(anchor_costs))]["q"].copy()

        if progress_cb is not None:
            progress_cb(i + 1, len(tcp_poses))

        if verbose:
            best = ranked[int(np.argmin([candidate["node_cost"] for candidate in ranked]))]
            path_ok = best["pos_err"] <= max_capture_pos_err
            status = "OK  " if path_ok else "FAIL"
            capture = "CAP" if best["capture_valid"] else "SKIP"
            collision = f" col={best['collision_count']}" if best["collision_count"] else ""
            print(
                f"  WP {i+1:3d}/{len(tcp_poses)}: "
                f"[{status}/{capture}] candidates={len(ranked):2d} "
                f"best_pos={best['pos_err']*1000:.2f}mm "
                f"best_look={best['look_deg']:.1f}deg "
                f"best_rot={np.rad2deg(best['rot_err']):.1f}deg"
                f"{collision}"
            )

    dp_costs = []
    parents = []
    first_cost = np.array(
        [
            candidate["node_cost"]
            + _joint_transition_cost(
                candidate["q"],
                q_start,
                max_joint_step,
                transition_weight,
                joint_step_weight,
                periodic_mask,
            )
            for candidate in candidate_sets[0]
        ],
        dtype=float,
    )
    dp_costs.append(first_cost)
    parents.append(np.full(len(candidate_sets[0]), -1, dtype=int))

    for i in range(1, len(candidate_sets)):
        prev_candidates = candidate_sets[i - 1]
        cur_candidates = candidate_sets[i]
        cur_cost = np.full(len(cur_candidates), np.inf, dtype=float)
        cur_parent = np.full(len(cur_candidates), -1, dtype=int)
        for cur_j, cur_candidate in enumerate(cur_candidates):
            transition_costs = np.array(
                [
                    _joint_transition_cost(
                        cur_candidate["q"],
                        prev_candidate["q"],
                        max_joint_step,
                        transition_weight,
                        joint_step_weight,
                        periodic_mask,
                    )
                    for prev_candidate in prev_candidates
                ],
                dtype=float,
            )
            total = dp_costs[i - 1] + transition_costs + cur_candidate["node_cost"]
            parent_j = int(np.argmin(total))
            cur_cost[cur_j] = float(total[parent_j])
            cur_parent[cur_j] = parent_j
        dp_costs.append(cur_cost)
        parents.append(cur_parent)

    choice = int(np.argmin(dp_costs[-1]))
    selected = [None] * len(candidate_sets)
    for i in range(len(candidate_sets) - 1, -1, -1):
        selected[i] = candidate_sets[i][choice]
        choice = int(parents[i][choice])
        if choice < 0 and i > 0:
            choice = 0

    Q = _unwrap_joint_path(np.array([candidate["q"] for candidate in selected], dtype=float), periodic_mask)
    flags = [
        bool(candidate["capture_valid"]) and int(candidate.get("collision_count", 0)) == 0
        for candidate in selected
    ]

    if verbose:
        dq = np.diff(Q, axis=0)
        if len(dq):
            print(
                "[IK] Path optimizer selected trajectory: "
                f"max ||dq||={np.max(np.linalg.norm(dq, axis=1)):.3f} rad, "
                f"max |dq_i|={np.max(np.abs(dq)):.3f} rad"
            )

    return Q, flags


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
    posture_bias=None,
    posture_weights=None,
    site_name="ee_site",
    max_joint_step=0.55,
    joint_step_weight=24.0,
    hard_max_joint_step=None,
    path_planning=True,
    candidate_limit=12,
    transition_weight=8.0,
    pose_position_scale=0.004,
    pose_look_scale=0.12,
    pose_rotation_scale=np.deg2rad(5.0),
    target_rotation_weight=8.0,
    target_rotation_tol=np.deg2rad(3.0),
    max_capture_look_deg=5.0,
    max_capture_pos_err=0.010,
    invalid_candidate_penalty=1.0e6,
    collision_penalty=0.0,
    collision_margin=0.0,
    progress_cb=None,
):
    rng = np.random.default_rng(rng_seed)
    limits = joint_limits(model, mujoco_module)
    qidx = joint_qpos_indices(model, mujoco_module)
    ndof = len(limits)
    q = np.zeros(ndof) if q_start is None else np.asarray(q_start, dtype=float).copy()
    Q = np.zeros((len(tcp_poses), ndof))
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

    if posture_weights is None:
        posture_weights_arr = np.ones(len(tcp_poses), dtype=float) if posture_bias else np.zeros(len(tcp_poses), dtype=float)
    else:
        posture_weights_arr = np.asarray(posture_weights, dtype=float)
        if posture_weights_arr.shape == ():
            posture_weights_arr = np.full(len(tcp_poses), float(posture_weights_arr))
        elif posture_weights_arr.shape != (len(tcp_poses),):
            raise ValueError(
                "posture_weights must be scalar or one weight per waypoint "
                f"with shape ({len(tcp_poses)},); got {posture_weights_arr.shape}."
            )

    def candidate_score(
        q_candidate,
        pose_score,
        q_reference,
        posture_weight,
        pos_err=0.0,
        look_err=0.0,
        rot_err=0.0,
        collision_count=0,
    ):
        score = _solution_score(
            q_candidate,
            pose_score,
            q_reference,
            limits,
            continuity_weight,
            limit_weight,
        )
        score += (float(rot_err) / max(float(pose_rotation_scale), 1e-9)) ** 2
        score += float(collision_penalty) * float(collision_count)
        score += float(joint_step_weight) * _joint_step_cost(
            q_candidate,
            q_reference,
            max_joint_step,
        )
        gate_cost, _, _ = _capture_gate_cost(
            pos_err,
            look_err,
            max_capture_pos_err,
            max_capture_look_deg,
            invalid_candidate_penalty,
        )
        score += gate_cost
        if posture_weight > 0.0 and posture_bias:
            score += posture_weight * _posture_bias_cost(
                model,
                data,
                mujoco_module,
                qidx,
                q_candidate,
                limits,
                posture_bias,
            )
        return score

    bias_q = posture_bias.get("q_ref") if posture_bias else None
    if bias_q is not None:
        bias_q = np.clip(np.asarray(bias_q, dtype=float), limits[:, 0], limits[:, 1])

    if path_planning and len(tcp_poses) > 1:
        return _solve_trajectory_path_dp(
            model,
            data,
            mujoco_module,
            tcp_poses,
            look_targets,
            q,
            retries,
            rng,
            verbose,
            axis_col,
            axis_sign,
            continuity_weight,
            limit_weight,
            method,
            posture_bias,
            posture_weights_arr,
            site_name,
            max_joint_step,
            joint_step_weight,
            candidate_limit,
            transition_weight,
            pose_position_scale,
            pose_look_scale,
            pose_rotation_scale,
            target_rotation_weight,
            target_rotation_tol,
            max_capture_look_deg,
            max_capture_pos_err,
            invalid_candidate_penalty,
            collision_penalty,
            collision_margin,
            progress_cb=progress_cb,
        )

    for i, (T, target_i) in enumerate(zip(tcp_poses, look_targets)):
        target_pos = T[:3, 3]
        target_R = T[:3, :3]
        posture_weight = float(posture_weights_arr[i])
        solver = solve_look_at_ik_optimized if method == "least_squares" else solve_look_at_ik
        best_q, best_ok, best_err = solver(
            model, data, mujoco_module, target_pos, target_i,
            q_init=q, q_ref=q, site_name=site_name, axis_col=axis_col, axis_sign=axis_sign,
            continuity_weight=continuity_weight, center_weight=limit_weight,
            target_R=target_R, rotation_weight=target_rotation_weight, tol_rot=target_rotation_tol,
        ) if solver is solve_look_at_ik_optimized else solver(
            model, data, mujoco_module, target_pos, target_i,
            q_init=q, q_ref=q, site_name=site_name, axis_col=axis_col, axis_sign=axis_sign,
            continuity_weight=continuity_weight, limit_weight=limit_weight,
            target_R=target_R, rotation_weight=target_rotation_weight, tol_rot=target_rotation_tol,
        )
        best_pos_err, best_look_err = _look_at_errors(
            model, data, mujoco_module, qidx, best_q, site_name, target_pos, target_i, axis_col, axis_sign
        )
        best_rot_err = _site_rotation_error(
            model, data, mujoco_module, qidx, best_q, site_name, target_R
        )
        best_collision = collision_contacts_for_q(
            model,
            data,
            mujoco_module,
            best_q,
            qidx=qidx,
            collision_margin=collision_margin,
            max_pairs=3,
        )
        best_err = best_pos_err
        best_score = candidate_score(
            best_q,
            best_pos_err + 2.0 * best_look_err,
            q,
            posture_weight,
            best_pos_err,
            best_look_err,
            best_rot_err,
            best_collision["count"],
        )

        if bias_q is not None and posture_weight > 0.0:
            for seed in (bias_q, 0.5 * (q + bias_q)):
                q_try, ok, err = solver(
                    model, data, mujoco_module, target_pos, target_i,
                    q_init=seed, q_ref=q, site_name=site_name, axis_col=axis_col, axis_sign=axis_sign,
                    continuity_weight=continuity_weight, center_weight=limit_weight,
                    target_R=target_R, rotation_weight=target_rotation_weight, tol_rot=target_rotation_tol,
                ) if solver is solve_look_at_ik_optimized else solver(
                    model, data, mujoco_module, target_pos, target_i,
                    q_init=seed, q_ref=q, site_name=site_name, axis_col=axis_col, axis_sign=axis_sign,
                    continuity_weight=continuity_weight, limit_weight=limit_weight,
                    target_R=target_R, rotation_weight=target_rotation_weight, tol_rot=target_rotation_tol,
                )
                pos_err, look_err = _look_at_errors(
                    model, data, mujoco_module, qidx, q_try, site_name, target_pos, target_i, axis_col, axis_sign
                )
                rot_err = _site_rotation_error(
                    model, data, mujoco_module, qidx, q_try, site_name, target_R
                )
                collision = collision_contacts_for_q(
                    model,
                    data,
                    mujoco_module,
                    q_try,
                    qidx=qidx,
                    collision_margin=collision_margin,
                    max_pairs=3,
                )
                score = candidate_score(
                    q_try,
                    pos_err + 2.0 * look_err,
                    q,
                    posture_weight,
                    pos_err,
                    look_err,
                    rot_err,
                    collision["count"],
                )
                if score < best_score:
                    best_q, best_ok, best_err, best_look_err, best_rot_err, best_collision, best_score = (
                        q_try, ok, pos_err, look_err, rot_err, collision, score
                    )

        for trial in range(retries):
            if best_ok and posture_weight <= 0.0:
                break
            if bias_q is not None and posture_weight > 0.0 and trial % 3 == 1:
                seed = bias_q + rng.uniform(-0.25, 0.25, ndof)
            elif trial % 3 == 0:
                seed = q + rng.uniform(-0.35, 0.35, ndof)
            else:
                seed = rng.uniform(limits[:, 0], limits[:, 1])
            seed = np.clip(seed, limits[:, 0], limits[:, 1])
            q_try, ok, err = solver(
                model, data, mujoco_module, target_pos, target_i,
                q_init=seed, q_ref=q, site_name=site_name, axis_col=axis_col, axis_sign=axis_sign,
                continuity_weight=continuity_weight, center_weight=limit_weight,
                target_R=target_R, rotation_weight=target_rotation_weight, tol_rot=target_rotation_tol,
            ) if solver is solve_look_at_ik_optimized else solver(
                model, data, mujoco_module, target_pos, target_i,
                q_init=seed, q_ref=q, site_name=site_name, axis_col=axis_col, axis_sign=axis_sign,
                continuity_weight=continuity_weight, limit_weight=limit_weight,
                target_R=target_R, rotation_weight=target_rotation_weight, tol_rot=target_rotation_tol,
            )
            pos_err, look_err = _look_at_errors(
                model, data, mujoco_module, qidx, q_try, site_name, target_pos, target_i, axis_col, axis_sign
            )
            rot_err = _site_rotation_error(
                model, data, mujoco_module, qidx, q_try, site_name, target_R
            )
            collision = collision_contacts_for_q(
                model,
                data,
                mujoco_module,
                q_try,
                qidx=qidx,
                collision_margin=collision_margin,
                max_pairs=3,
            )
            score = candidate_score(
                q_try,
                pos_err + 2.0 * look_err,
                q,
                posture_weight,
                pos_err,
                look_err,
                rot_err,
                collision["count"],
            )
            if score < best_score:
                best_q, best_ok, best_err, best_look_err, best_rot_err, best_collision, best_score = (
                    q_try, ok, pos_err, look_err, rot_err, collision, score
                )

        if i > 0 and hard_max_joint_step is not None and hard_max_joint_step > 0.0:
            delta = best_q - q
            if np.max(np.abs(delta)) > hard_max_joint_step:
                best_q = q + np.clip(delta, -hard_max_joint_step, hard_max_joint_step)
                best_err, best_look_err = _look_at_errors(
                    model,
                    data,
                    mujoco_module,
                    qidx,
                    best_q,
                    site_name,
                    target_pos,
                    target_i,
                    axis_col,
                    axis_sign,
                )
                best_rot_err = _site_rotation_error(
                    model, data, mujoco_module, qidx, best_q, site_name, target_R
                )
                best_collision = collision_contacts_for_q(
                    model,
                    data,
                    mujoco_module,
                    best_q,
                    qidx=qidx,
                    collision_margin=collision_margin,
                    max_pairs=3,
                )
                best_ok = False

        Q[i] = best_q
        capture_valid, best_look_deg = _capture_gate(
            best_err,
            best_look_err,
            max_capture_pos_err,
            max_capture_look_deg,
        )
        flags.append(bool(capture_valid))
        if best_collision["count"] > 0:
            flags[-1] = False
        q = best_q.copy()

        if progress_cb is not None:
            progress_cb(i + 1, len(tcp_poses))

        if verbose:
            status = "OK  " if best_ok else "FAIL"
            capture = "CAP" if capture_valid else "SKIP"
            collision = f" col={best_collision['count']}" if best_collision["count"] else ""
            print(
                f"  WP {i+1:3d}/{len(tcp_poses)}: [{status}/{capture}] "
                f"pos_err={best_err*1000:.2f}mm look={best_look_deg:.1f}deg "
                f"rot={np.rad2deg(best_rot_err):.1f}deg"
                f"{collision}"
            )

    return Q, flags
