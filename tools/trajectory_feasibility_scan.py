#!/usr/bin/env python3
"""Scan whether the robot can keep the D405 looking at the pipe/flange seam."""

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

from control.ik_solver import (
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
    TRAJECTORY_CENTER,
    TRAJECTORY_RADIUS,
    camera_poses_to_tcp_poses,
    right_posture_weights,
)
from trajectory.circle import segmented_circle_trajectory


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


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


def limit_metrics(Q: np.ndarray, limits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower_margin = Q - limits[:, 0]
    upper_margin = limits[:, 1] - Q
    margins = np.minimum(lower_margin, upper_margin)
    closest = np.argmin(margins, axis=1) + 1
    return np.min(margins, axis=1), closest


def build_traj(x_offset: float, radius: float, waypoints: int) -> dict:
    traj = segmented_circle_trajectory(
        center=FLANGE_CENTER + np.array([x_offset, 0.0, 0.0]),
        radius=radius,
        segment_duration=9.0,
        dt=0.02,
        orientation_target=FLANGE_CENTER,
        target_radius=SEAM_TARGET_RADIUS,
        feasible_only=False,
    )
    return sample_trajectory(traj, waypoints)


def valid_groups(valid: np.ndarray, segment_ids: np.ndarray, angles: np.ndarray) -> list[dict]:
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return []

    groups = []
    start = int(idx[0])
    prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if cur != prev + 1 or int(segment_ids[cur]) != int(segment_ids[prev]):
            groups.append(group_info(start, prev, segment_ids, angles))
            start = cur
        prev = cur
    groups.append(group_info(start, prev, segment_ids, angles))
    return groups


def group_info(start: int, end: int, segment_ids: np.ndarray, angles: np.ndarray) -> dict:
    phi0 = float(np.rad2deg(angles[start]))
    phif = float(np.rad2deg(angles[end]))
    return {
        "start_index": start,
        "end_index": end,
        "count": end - start + 1,
        "segment_id": int(segment_ids[start]),
        "phi_start_deg": phi0,
        "phi_end_deg": phif,
        "arc_span_deg": abs(phif - phi0),
    }


def failure_reason(
    look_deg: float,
    pos_err_m: float,
    margin: float,
    look_threshold: float,
    pos_threshold_m: float,
    margin_threshold: float,
) -> str:
    reasons = []
    if look_deg > look_threshold:
        reasons.append("look")
    if pos_err_m > pos_threshold_m:
        reasons.append("position")
    if margin < margin_threshold:
        reasons.append("joint_limit")
    return "+".join(reasons) if reasons else ""


def scan_config(
    model,
    data,
    x_offset: float,
    radius: float,
    waypoints: int,
    retries: int,
    rng_seed: int,
    look_threshold: float,
    pos_threshold_m: float,
    margin_threshold: float,
    verbose_ik: bool,
) -> dict:
    traj = build_traj(x_offset, radius, waypoints)
    tcp_poses = camera_poses_to_tcp_poses(traj["poses"])
    posture_weights = right_posture_weights(traj, len(tcp_poses))
    Q, _ = solve_trajectory(
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
        max_capture_look_deg=look_threshold,
        max_capture_pos_err=pos_threshold_m,
        site_name=CAMERA_SITE_NAME,
    )
    metrics = evaluate_look_at_trajectory(
        model,
        data,
        mujoco,
        Q,
        tcp_poses,
        traj["targets"],
        axis_col=EE_LOOK_AXIS_COL,
        axis_sign=EE_LOOK_AXIS_SIGN,
        site_name=CAMERA_SITE_NAME,
        max_pos_err=pos_threshold_m,
        max_look_deg=look_threshold,
    )

    limits = joint_limits(model, mujoco)
    margin_min, closest_joint = limit_metrics(Q, limits)
    dq = np.diff(Q, axis=0)
    dq_norm = np.zeros(len(Q), dtype=float)
    dq_abs = np.zeros(len(Q), dtype=float)
    if len(dq):
        dq_norm[1:] = np.linalg.norm(dq, axis=1)
        dq_abs[1:] = np.max(np.abs(dq), axis=1)

    valid = metrics["capture_valid"]
    segment_ids = np.asarray(traj.get("segment_id", np.ones(len(Q), dtype=int)), dtype=int)
    groups = valid_groups(valid, segment_ids, traj["angles"])

    rows = []
    for i, q in enumerate(Q):
        reason = failure_reason(
            float(metrics["look_deg"][i]),
            float(metrics["pos_err"][i]),
            float(margin_min[i]),
            look_threshold,
            pos_threshold_m,
            margin_threshold,
        )
        row = {
            "config_id": "",
            "index": i,
            "x_offset_m": float(x_offset),
            "radius_m": float(radius),
            "segment_id": int(segment_ids[i]),
            "segment_name": str(traj.get("segment_name", np.asarray([""] * len(Q)))[i]),
            "phi_deg": float(np.rad2deg(traj["angles"][i])),
            "capture_valid": int(valid[i]),
            "failure_reason": reason,
            "look_error_deg": float(metrics["look_deg"][i]),
            "position_error_mm": float(metrics["pos_err"][i]) * 1000.0,
            "joint_limit_margin_min_rad": float(margin_min[i]),
            "closest_limit_joint": int(closest_joint[i]),
            "dq_norm": float(dq_norm[i]),
            "dq_abs_max": float(dq_abs[i]),
            "camera_x": float(traj["positions"][i][0]),
            "camera_y": float(traj["positions"][i][1]),
            "camera_z": float(traj["positions"][i][2]),
            "target_x": float(traj["targets"][i][0]),
            "target_y": float(traj["targets"][i][1]),
            "target_z": float(traj["targets"][i][2]),
        }
        row.update({f"q{j + 1}": float(q[j]) for j in range(Q.shape[1])})
        rows.append(row)

    valid_count = int(np.count_nonzero(valid))
    best_group = max(groups, key=lambda item: item["count"]) if groups else None
    summary = {
        "config_id": "",
        "x_offset_m": float(x_offset),
        "radius_m": float(radius),
        "waypoints": int(len(Q)),
        "valid_count": valid_count,
        "valid_ratio": float(valid_count / max(len(Q), 1)),
        "valid_groups": int(len(groups)),
        "best_group_count": 0 if best_group is None else int(best_group["count"]),
        "best_group_segment": "" if best_group is None else int(best_group["segment_id"]),
        "best_group_phi_start_deg": "" if best_group is None else float(best_group["phi_start_deg"]),
        "best_group_phi_end_deg": "" if best_group is None else float(best_group["phi_end_deg"]),
        "look_error_deg_min": float(np.min(metrics["look_deg"])),
        "look_error_deg_median": float(np.median(metrics["look_deg"])),
        "look_error_deg_max": float(np.max(metrics["look_deg"])),
        "position_error_mm_median": float(np.median(metrics["pos_err"]) * 1000.0),
        "position_error_mm_max": float(np.max(metrics["pos_err"]) * 1000.0),
        "joint_limit_margin_min_rad": float(np.min(margin_min)),
        "near_limit_count": int(np.count_nonzero(margin_min < margin_threshold)),
        "dq_abs_max": float(np.max(dq_abs)),
        "dq_norm_max": float(np.max(dq_norm)),
        "usable_arcs": format_groups(groups),
    }
    return {"summary": summary, "rows": rows, "groups": groups}


def format_groups(groups: list[dict]) -> str:
    if not groups:
        return ""
    parts = []
    for group in groups:
        parts.append(
            f"S{group['segment_id']}:{group['phi_start_deg']:.1f}->{group['phi_end_deg']:.1f}"
            f"({group['count']})"
        )
    return "; ".join(parts)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_report(summaries: list[dict], detail_csv: Path, summary_csv: Path) -> None:
    summaries = sorted(
        summaries,
        key=lambda row: (
            -row["valid_ratio"],
            row["look_error_deg_median"],
            -row["joint_limit_margin_min_rad"],
            row["dq_abs_max"],
        ),
    )
    print(f"[scan] wrote detail CSV : {detail_csv}")
    print(f"[scan] wrote summary CSV: {summary_csv}")
    print("\n[scan] best configs")
    print("rank config valid look_med look_max min_margin max_dq x_offset radius usable_arcs")
    for rank, row in enumerate(summaries[:8], start=1):
        print(
            f"{rank:>4} {row['config_id']:>6} "
            f"{row['valid_count']:>3}/{row['waypoints']:<3} "
            f"{row['look_error_deg_median']:>8.2f} "
            f"{row['look_error_deg_max']:>8.2f} "
            f"{row['joint_limit_margin_min_rad']:>10.3f} "
            f"{row['dq_abs_max']:>7.2f} "
            f"{row['x_offset_m']:>8.3f} "
            f"{row['radius_m']:>6.3f} "
            f"{row['usable_arcs']}"
        )

    current = next((row for row in summaries if row["config_id"] == "cfg000"), None)
    if current is not None:
        print("\n[scan] current/default config")
        print(
            f"valid={current['valid_count']}/{current['waypoints']} "
            f"median_look={current['look_error_deg_median']:.2f}deg "
            f"max_look={current['look_error_deg_max']:.2f}deg "
            f"min_margin={current['joint_limit_margin_min_rad']:.3f}rad"
        )
        print(f"usable arcs: {current['usable_arcs'] or '(none)'}")


def make_configs(args) -> list[tuple[float, float]]:
    if args.quick_grid:
        default_config = (
            float(TRAJECTORY_CENTER[0] - FLANGE_CENTER[0]),
            float(TRAJECTORY_RADIUS),
        )
        configs = [
            default_config,
            (-0.105, 0.182),
            (-0.150, 0.140),
            (-0.180, 0.120),
        ]
        unique_configs = []
        for config in configs:
            if not any(np.allclose(config, other) for other in unique_configs):
                unique_configs.append(config)
        return unique_configs
    x_offsets = parse_float_list(args.x_offsets)
    radii = parse_float_list(args.radii)
    return [(x_offset, radius) for x_offset in x_offsets for radius in radii]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE_XML)
    parser.add_argument("--waypoints", type=int, default=48)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--rng-seed", type=int, default=7)
    parser.add_argument("--x-offsets", default=f"{float(TRAJECTORY_CENTER[0] - FLANGE_CENTER[0]):.6f}")
    parser.add_argument("--radii", default=f"{float(TRAJECTORY_RADIUS):.6f}")
    parser.add_argument("--quick-grid", action="store_true", help="Scan a small set of candidate x/radius pairs.")
    parser.add_argument("--look-deg", type=float, default=CAPTURE_MAX_LOOK_DEG)
    parser.add_argument("--pos-mm", type=float, default=CAPTURE_MAX_POS_ERR * 1000.0)
    parser.add_argument("--limit-margin", type=float, default=0.05)
    parser.add_argument("--detail-csv", default="outputs/diagnostics/trajectory_feasibility_detail.csv")
    parser.add_argument("--summary-csv", default="outputs/diagnostics/trajectory_feasibility_summary.csv")
    parser.add_argument("--verbose-ik", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    configs = make_configs(args)

    detail_rows = []
    summaries = []
    for config_i, (x_offset, radius) in enumerate(configs):
        config_id = f"cfg{config_i:03d}"
        print(
            f"[scan] {config_id}: x_offset={x_offset:.3f}m radius={radius:.3f}m "
            f"waypoints={args.waypoints} retries={args.retries}"
        )
        result = scan_config(
            model=model,
            data=data,
            x_offset=x_offset,
            radius=radius,
            waypoints=args.waypoints,
            retries=args.retries,
            rng_seed=args.rng_seed + config_i,
            look_threshold=args.look_deg,
            pos_threshold_m=args.pos_mm / 1000.0,
            margin_threshold=args.limit_margin,
            verbose_ik=args.verbose_ik,
        )
        result["summary"]["config_id"] = config_id
        for row in result["rows"]:
            row["config_id"] = config_id
        summaries.append(result["summary"])
        detail_rows.extend(result["rows"])

    detail_csv = Path(args.detail_csv)
    summary_csv = Path(args.summary_csv)
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summaries)
    print_report(summaries, detail_csv, summary_csv)


if __name__ == "__main__":
    main()
