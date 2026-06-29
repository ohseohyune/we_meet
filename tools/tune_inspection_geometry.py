#!/usr/bin/env python3
"""Tune pipe/flange height and inspection trajectory geometry for IK quality."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.franka_ik_solver import (
    evaluate_collision_trajectory,
    evaluate_look_at_trajectory,
    joint_limits,
    set_arm_qpos,
    solve_trajectory,
)
from mujoco_viewer import (
    CAPTURE_MAX_LOOK_DEG,
    CAPTURE_MAX_POS_ERR,
    CAMERA_SITE_NAME,
    EE_LOOK_AXIS_COL,
    EE_LOOK_AXIS_SIGN,
    FLANGE_CENTER,
    RIGHT_BIASED_READY,
    RIGHT_POSTURE_BIAS,
    SCENE_XML,
    SEAM_TARGET_RADIUS,
    camera_poses_to_tcp_poses,
    right_posture_weights,
)
from trajectory.circle import segmented_circle_trajectory


PIPE_RADIUS = 0.0605 / 2.0
SUPPORT_SADDLE_HALF_HEIGHT = 0.0120
SUPPORT_POST_BOTTOM_Z = 0.0300


def parse_values(text: str) -> list[float]:
    """Parse comma values or inclusive start:stop:step ranges."""
    values: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            values.append(float(part))
            continue
        pieces = [float(item) for item in part.split(":")]
        if len(pieces) != 3:
            raise ValueError(f"Range must be start:stop:step, got {part!r}")
        start, stop, step = pieces
        if step == 0.0:
            raise ValueError("Range step must be non-zero.")
        n = int(np.floor((stop - start) / step + 1e-9)) + 1
        values.extend(float(start + i * step) for i in range(max(0, n)))
    return sorted(set(round(v, 10) for v in values))


def sample_trajectory(traj: dict, waypoints: int) -> dict:
    if len(traj["positions"]) <= waypoints:
        return traj
    sample_idx = np.linspace(0, len(traj["positions"]) - 1, waypoints).round().astype(int)
    sampled = {}
    for key, value in traj.items():
        if isinstance(value, np.ndarray) and len(value) == len(traj["positions"]):
            sampled[key] = value[sample_idx]
        else:
            sampled[key] = value
    return sampled


def limit_margin(Q: np.ndarray, limits: np.ndarray) -> np.ndarray:
    lower_margin = Q - limits[:, 0]
    upper_margin = limits[:, 1] - Q
    return np.min(np.minimum(lower_margin, upper_margin), axis=1)


def set_scene_height(model, height: float) -> None:
    pipe_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pipe_flange_assembly")
    if pipe_body >= 0:
        model.body_pos[pipe_body, 2] = float(height)

    saddle_z = float(height) - PIPE_RADIUS - SUPPORT_SADDLE_HALF_HEIGHT
    post_top = saddle_z - SUPPORT_SADDLE_HALF_HEIGHT
    post_center = 0.5 * (SUPPORT_POST_BOTTOM_Z + post_top)
    post_halfheight = max(0.001, 0.5 * (post_top - SUPPORT_POST_BOTTOM_Z))

    support_post = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "support_post")
    if support_post >= 0:
        model.geom_pos[support_post, 2] = post_center
        model.geom_size[support_post, 1] = post_halfheight

    support_saddle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "support_saddle")
    if support_saddle >= 0:
        model.geom_pos[support_saddle, 2] = saddle_z


def build_traj(height: float, x_distance: float, diameter: float, waypoints: int) -> dict:
    flange_center = np.array([FLANGE_CENTER[0], 0.0, height], dtype=float)
    radius = 0.5 * float(diameter)
    center = flange_center + np.array([-float(x_distance), 0.0, 0.0], dtype=float)
    traj = segmented_circle_trajectory(
        center=center,
        radius=radius,
        segment_duration=9.0,
        dt=0.02,
        orientation_target=flange_center,
        target_radius=SEAM_TARGET_RADIUS,
        feasible_only=False,
    )
    return sample_trajectory(traj, waypoints)


def evaluate_config(
    model,
    data,
    height: float,
    x_distance: float,
    diameter: float,
    waypoints: int,
    retries: int,
    rng_seed: int,
    look_deg: float,
    pos_m: float,
    collision_margin: float,
    verbose_ik: bool,
) -> dict:
    set_scene_height(model, height)
    mujoco.mj_forward(model, data)

    traj = build_traj(height, x_distance, diameter, waypoints)
    tcp_poses = camera_poses_to_tcp_poses(traj["poses"])
    posture_weights = right_posture_weights(traj, len(tcp_poses))
    Q, flags = solve_trajectory(
        model,
        data,
        mujoco,
        tcp_poses,
        look_target=traj["targets"],
        q_start=RIGHT_BIASED_READY.copy(),
        retries=retries,
        rng_seed=rng_seed,
        verbose=verbose_ik,
        axis_col=EE_LOOK_AXIS_COL,
        axis_sign=EE_LOOK_AXIS_SIGN,
        posture_bias=RIGHT_POSTURE_BIAS,
        posture_weights=posture_weights,
        max_capture_look_deg=look_deg,
        max_capture_pos_err=pos_m,
        collision_penalty=8.0,
        collision_margin=collision_margin,
        site_name=CAMERA_SITE_NAME,
    )
    flags = np.asarray(flags, dtype=bool)

    capture = evaluate_look_at_trajectory(
        model,
        data,
        mujoco,
        Q,
        tcp_poses,
        traj["targets"],
        axis_col=EE_LOOK_AXIS_COL,
        axis_sign=EE_LOOK_AXIS_SIGN,
        site_name=CAMERA_SITE_NAME,
        max_pos_err=pos_m,
        max_look_deg=look_deg,
    )
    collision = evaluate_collision_trajectory(
        model,
        data,
        mujoco,
        Q,
        collision_margin=collision_margin,
        max_pairs=1,
    )
    margins = limit_margin(Q, joint_limits(model, mujoco))

    capture_valid = np.asarray(capture["capture_valid"], dtype=bool)
    collision_free = np.asarray(collision["collision_free"], dtype=bool)
    ok = flags & capture_valid & collision_free
    pos_mm = np.asarray(capture["pos_err"], dtype=float) * 1000.0
    look = np.asarray(capture["look_deg"], dtype=float)

    dq_abs = 0.0
    if len(Q) > 1:
        dq_abs = float(np.max(np.abs(np.diff(Q, axis=0))))

    valid_count = int(np.count_nonzero(ok))
    flag_count = int(np.count_nonzero(flags))
    capture_count = int(np.count_nonzero(capture_valid))
    collision_free_count = int(np.count_nonzero(collision_free))
    collision_count = int(np.count_nonzero(~collision_free))
    waypoint_count = int(len(Q))

    # Hard validity dominates, then tracking error and smoothness decide ties.
    score = (
        100000.0 * (waypoint_count - valid_count)
        + 1000.0 * (waypoint_count - flag_count)
        + 150.0 * float(np.median(look))
        + 30.0 * float(np.max(look))
        + 8.0 * float(np.median(pos_mm))
        + 2.0 * float(np.max(pos_mm))
        + 250.0 * collision_count
        + 20.0 * dq_abs
        - 20.0 * float(np.min(margins))
    )

    return {
        "height_m": float(height),
        "x_distance_m": float(x_distance),
        "trajectory_diameter_m": float(diameter),
        "trajectory_radius_m": float(0.5 * diameter),
        "waypoints": waypoint_count,
        "score": float(score),
        "ok_count": valid_count,
        "ok_ratio": float(valid_count / max(waypoint_count, 1)),
        "ik_flag_count": flag_count,
        "capture_valid_count": capture_count,
        "collision_free_count": collision_free_count,
        "collision_count": collision_count,
        "look_error_deg_median": float(np.median(look)),
        "look_error_deg_max": float(np.max(look)),
        "position_error_mm_median": float(np.median(pos_mm)),
        "position_error_mm_max": float(np.max(pos_mm)),
        "joint_limit_margin_min_rad": float(np.min(margins)),
        "dq_abs_max_rad": dq_abs,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sorted_results(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            -r["ok_ratio"],
            -r["ik_flag_count"],
            -r["capture_valid_count"],
            -r["collision_free_count"],
            r["score"],
            r["look_error_deg_median"],
            r["position_error_mm_median"],
        ),
    )


def print_report(rows: list[dict], csv_path: Path, top: int) -> None:
    ranked = sorted_results(rows)
    print(f"[tune] wrote CSV: {csv_path}")
    print("[tune] best configs")
    print("rank ok ik cap coll_free look_med pos_med height x_dist diameter score")
    for i, row in enumerate(ranked[:top], start=1):
        print(
            f"{i:>4} "
            f"{row['ok_count']:>2}/{row['waypoints']:<2} "
            f"{row['ik_flag_count']:>2}/{row['waypoints']:<2} "
            f"{row['capture_valid_count']:>2}/{row['waypoints']:<2} "
            f"{row['collision_free_count']:>2}/{row['waypoints']:<2} "
            f"{row['look_error_deg_median']:>8.2f} "
            f"{row['position_error_mm_median']:>7.2f} "
            f"{row['height_m']:>6.3f} "
            f"{row['x_distance_m']:>6.3f} "
            f"{row['trajectory_diameter_m']:>8.3f} "
            f"{row['score']:>10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE_XML)
    parser.add_argument("--heights", default="0.52:0.82:0.05")
    parser.add_argument("--x-distances", default="0.06:0.18:0.02")
    parser.add_argument("--diameters", default="0.24:0.54:0.05")
    parser.add_argument("--waypoints", type=int, default=16)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--look-deg", type=float, default=CAPTURE_MAX_LOOK_DEG)
    parser.add_argument("--pos-mm", type=float, default=CAPTURE_MAX_POS_ERR * 1000.0)
    parser.add_argument("--collision-margin", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--csv", default="outputs/diagnostics/inspection_geometry_tuning.csv")
    parser.add_argument("--verbose-ik", action="store_true")
    args = parser.parse_args()

    heights = parse_values(args.heights)
    x_distances = parse_values(args.x_distances)
    diameters = parse_values(args.diameters)
    configs = [
        (height, x_distance, diameter)
        for height in heights
        for x_distance in x_distances
        for diameter in diameters
    ]

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    rows = []
    total = len(configs)
    for idx, (height, x_distance, diameter) in enumerate(configs, start=1):
        print(
            f"[tune] {idx:03d}/{total:03d} "
            f"h={height:.3f} x_dist={x_distance:.3f} dia={diameter:.3f}"
        )
        row = evaluate_config(
            model=model,
            data=data,
            height=height,
            x_distance=x_distance,
            diameter=diameter,
            waypoints=args.waypoints,
            retries=args.retries,
            rng_seed=args.rng_seed + idx,
            look_deg=args.look_deg,
            pos_m=args.pos_mm / 1000.0,
            collision_margin=args.collision_margin,
            verbose_ik=args.verbose_ik,
        )
        rows.append(row)

    csv_path = Path(args.csv)
    write_csv(csv_path, rows)
    print_report(rows, csv_path, args.top)


if __name__ == "__main__":
    main()
