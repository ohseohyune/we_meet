#!/usr/bin/env python3
"""Export reference-vs-current tracking error plots for presentation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco

from control.ik_solver import set_arm_qpos, site_pose
from mujoco_viewer import (
    CAMERA_SITE_NAME,
    DEFAULT_IK_RETRIES,
    SCENE_XML,
    compute_ik_trajectory,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "presentation" / "tracking_error"
ACCENT = "#cf1b68"
MAROON = "#6b1730"
BLUE = "#1f77b4"
TEAL = "#008b8b"
ORANGE = "#dd8a16"
GREY = "#767676"
GRID = "#d9d2c3"


def _setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=15, fontweight="bold", color="#111111", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8a8174")
    ax.spines["bottom"].set_color("#8a8174")
    ax.tick_params(colors="#333333", labelsize=10)


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(
            OUT_DIR / f"{name}.{ext}",
            dpi=240,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def _rotation_error_deg(R_ref: np.ndarray, R_cur: np.ndarray) -> float:
    R_err = np.asarray(R_ref, dtype=float).T @ np.asarray(R_cur, dtype=float)
    cos_angle = (np.trace(R_err) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_angle)))


def _segment_boundaries(time: np.ndarray, traj_info: dict) -> list[float]:
    segment_ids = np.asarray(traj_info.get("segment_id", []), dtype=int)
    if len(segment_ids) != len(time):
        return []
    idx = np.where(np.diff(segment_ids) != 0)[0] + 1
    return [float(time[i]) for i in idx]


def _draw_boundaries(ax, boundaries: list[float]) -> None:
    for boundary in boundaries:
        ax.axvline(boundary, color=GREY, linewidth=1.0, linestyle="--", alpha=0.55)


def _collect_actual_camera_poses(model, data, q_values: np.ndarray) -> np.ndarray:
    actual = []
    for q in np.asarray(q_values, dtype=float):
        set_arm_qpos(model, data, mujoco, q)
        actual.append(site_pose(model, data, mujoco, CAMERA_SITE_NAME))
    return np.asarray(actual)


def _plot_position_overlay(
    time: np.ndarray,
    ref_pos: np.ndarray,
    cur_pos: np.ndarray,
    boundaries: list[float],
) -> None:
    labels = ("x", "y", "z")
    colors = (BLUE, TEAL, ACCENT)
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 7.0), sharex=True)
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, ref_pos[:, i], color=color, linewidth=2.4, label=f"reference {label}")
        ax.plot(time, cur_pos[:, i], color=MAROON, linewidth=1.8, linestyle="--", label=f"current {label}")
        _draw_boundaries(ax, boundaries)
        _setup_axes(ax, f"camera {label}: reference vs current", "time [s]", "position [m]")
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout(pad=1.4)
    _save(fig, "camera_position_reference_current")


def _plot_position_error(time: np.ndarray, pos_err: np.ndarray, boundaries: list[float]) -> None:
    err_mm = pos_err * 1000.0
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(time, err_mm[:, 0], color=BLUE, linewidth=2.0, label="x error")
    ax.plot(time, err_mm[:, 1], color=TEAL, linewidth=2.0, label="y error")
    ax.plot(time, err_mm[:, 2], color=ACCENT, linewidth=2.0, label="z error")
    ax.plot(time, np.linalg.norm(err_mm, axis=1), color=MAROON, linewidth=2.8, label="norm")
    _draw_boundaries(ax, boundaries)
    _setup_axes(ax, "camera position error", "time [s]", "error [mm]")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper center")
    _save(fig, "camera_position_error")


def _plot_orientation_error(time: np.ndarray, rot_err_deg: np.ndarray, boundaries: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(time, rot_err_deg, color=ACCENT, linewidth=2.8)
    _draw_boundaries(ax, boundaries)
    _setup_axes(ax, "camera orientation error", "time [s]", "rotation error [deg]")
    _save(fig, "camera_orientation_error")


def _plot_summary(
    time: np.ndarray,
    ref_pos: np.ndarray,
    cur_pos: np.ndarray,
    pos_err: np.ndarray,
    rot_err_deg: np.ndarray,
    boundaries: list[float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.0))

    ax = axes[0, 0]
    for idx, (label, color) in enumerate(zip(("x", "y", "z"), (BLUE, TEAL, ACCENT))):
        ax.plot(time, ref_pos[:, idx], color=color, linewidth=2.2, label=f"ref {label}")
        ax.plot(time, cur_pos[:, idx], color=color, linewidth=1.4, linestyle="--", alpha=0.7, label=f"cur {label}")
    _draw_boundaries(ax, boundaries)
    _setup_axes(ax, "position reference vs current", "time [s]", "position [m]")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")

    ax = axes[0, 1]
    ax.plot(time, np.linalg.norm(pos_err, axis=1) * 1000.0, color=MAROON, linewidth=2.8)
    _draw_boundaries(ax, boundaries)
    _setup_axes(ax, "position error norm", "time [s]", "error [mm]")

    ax = axes[1, 0]
    err_mm = pos_err * 1000.0
    for idx, (label, color) in enumerate(zip(("x", "y", "z"), (BLUE, TEAL, ACCENT))):
        ax.plot(time, err_mm[:, idx], color=color, linewidth=2.1, label=f"{label} error")
    _draw_boundaries(ax, boundaries)
    _setup_axes(ax, "position component error", "time [s]", "error [mm]")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")

    ax = axes[1, 1]
    ax.plot(time, rot_err_deg, color=ACCENT, linewidth=2.8)
    _draw_boundaries(ax, boundaries)
    _setup_axes(ax, "orientation error", "time [s]", "error [deg]")

    fig.tight_layout(pad=2.0)
    _save(fig, "tracking_error_summary_2x2")


def _write_csv(
    time: np.ndarray,
    angles: np.ndarray,
    flags: np.ndarray,
    ref_poses: np.ndarray,
    cur_poses: np.ndarray,
    pos_err: np.ndarray,
    rot_err_deg: np.ndarray,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pos_err_norm = np.linalg.norm(pos_err, axis=1)
    metrics = {
        "waypoints": len(time),
        "ik_success": int(np.count_nonzero(flags)),
        "ik_total": len(flags),
        "max_position_error_mm": float(np.max(pos_err_norm) * 1000.0),
        "mean_position_error_mm": float(np.mean(pos_err_norm) * 1000.0),
        "median_position_error_mm": float(np.median(pos_err_norm) * 1000.0),
        "max_orientation_error_deg": float(np.max(rot_err_deg)),
        "mean_orientation_error_deg": float(np.mean(rot_err_deg)),
        "median_orientation_error_deg": float(np.median(rot_err_deg)),
    }
    with open(OUT_DIR / "tracking_error_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    fieldnames = [
        "index",
        "time_s",
        "angle_deg",
        "ik_success",
        "ref_x",
        "ref_y",
        "ref_z",
        "cur_x",
        "cur_y",
        "cur_z",
        "err_x_mm",
        "err_y_mm",
        "err_z_mm",
        "err_norm_mm",
        "rot_err_deg",
    ]
    with open(OUT_DIR / "tracking_error_timeseries.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(time)):
            writer.writerow(
                {
                    "index": i,
                    "time_s": float(time[i]),
                    "angle_deg": float(np.rad2deg(angles[i])),
                    "ik_success": int(flags[i]),
                    "ref_x": float(ref_poses[i, 0, 3]),
                    "ref_y": float(ref_poses[i, 1, 3]),
                    "ref_z": float(ref_poses[i, 2, 3]),
                    "cur_x": float(cur_poses[i, 0, 3]),
                    "cur_y": float(cur_poses[i, 1, 3]),
                    "cur_z": float(cur_poses[i, 2, 3]),
                    "err_x_mm": float(pos_err[i, 0] * 1000.0),
                    "err_y_mm": float(pos_err[i, 1] * 1000.0),
                    "err_z_mm": float(pos_err[i, 2] * 1000.0),
                    "err_norm_mm": float(pos_err_norm[i] * 1000.0),
                    "rot_err_deg": float(rot_err_deg[i]),
                }
            )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    Q, flags, time, angles, _positions, _rotations, camera_poses, _tcp_poses, traj_info = (
        compute_ik_trajectory(
            model,
            data,
            verbose=False,
            retries=DEFAULT_IK_RETRIES,
            max_waypoints=36,
        )
    )

    ref_poses = np.asarray(camera_poses, dtype=float)
    cur_poses = _collect_actual_camera_poses(model, data, Q)
    time = np.asarray(time, dtype=float)
    angles = np.asarray(angles, dtype=float)
    flags = np.asarray(flags, dtype=bool)

    ref_pos = ref_poses[:, :3, 3]
    cur_pos = cur_poses[:, :3, 3]
    pos_err = cur_pos - ref_pos
    rot_err_deg = np.array(
        [_rotation_error_deg(ref_poses[i, :3, :3], cur_poses[i, :3, :3]) for i in range(len(ref_poses))],
        dtype=float,
    )
    boundaries = _segment_boundaries(time, traj_info)

    _plot_position_overlay(time, ref_pos, cur_pos, boundaries)
    _plot_position_error(time, pos_err, boundaries)
    _plot_orientation_error(time, rot_err_deg, boundaries)
    _plot_summary(time, ref_pos, cur_pos, pos_err, rot_err_deg, boundaries)
    _write_csv(time, angles, flags, ref_poses, cur_poses, pos_err, rot_err_deg)

    pos_err_norm_mm = np.linalg.norm(pos_err, axis=1) * 1000.0
    print(f"Wrote tracking error plots to {OUT_DIR}")
    print(f"IK success: {int(np.count_nonzero(flags))}/{len(flags)}")
    print(
        "Position error: "
        f"mean={np.mean(pos_err_norm_mm):.4f} mm, "
        f"median={np.median(pos_err_norm_mm):.4f} mm, "
        f"max={np.max(pos_err_norm_mm):.4f} mm"
    )
    print(
        "Orientation error: "
        f"mean={np.mean(rot_err_deg):.4f} deg, "
        f"median={np.median(rot_err_deg):.4f} deg, "
        f"max={np.max(rot_err_deg):.4f} deg"
    )


if __name__ == "__main__":
    main()
