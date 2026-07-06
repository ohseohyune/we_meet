#!/usr/bin/env python3
"""Export quantitative evaluation metrics as CSV and presentation table images."""

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

from control.ik_solver import joint_limits
from mujoco_viewer import DEFAULT_IK_RETRIES, SCENE_XML, compute_ik_trajectory


OUT_DIR = PROJECT_ROOT / "outputs" / "presentation" / "quantitative_metrics"
HEADER = "#68152b"
ROW_ALT = "#f7f4ee"
GRID = "#e0d7c7"
TEXT = "#111111"
ACCENT = "#cf1b68"


def _gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, time, axis=0, edge_order=1)


def _joint_limit_margin(q: np.ndarray, limits: np.ndarray) -> np.ndarray:
    lower_margin = q - limits[:, 0]
    upper_margin = limits[:, 1] - q
    return np.minimum(lower_margin, upper_margin)


def _fmt_count(ok: int, total: int) -> str:
    return f"{ok}/{total}"


def _fmt_float(value: float, digits: int = 3) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def _write_csv(metrics: list[tuple[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "quantitative_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics)


def _draw_single_table(metrics: list[tuple[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [[name, value] for name, value in metrics]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        colWidths=[0.76, 0.24],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.65)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor(HEADER)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor(ROW_ALT if r % 2 == 0 else "white")
            cell.set_text_props(color=TEXT)
            if c == 1:
                cell.set_text_props(color=ACCENT, weight="bold")

    fig.savefig(OUT_DIR / "quantitative_metrics_table.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "quantitative_metrics_table.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_two_column_table(left: list[tuple[str, str]], right: list[tuple[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6))
    for ax, metrics in zip(axes, (left, right)):
        ax.axis("off")
        table = ax.table(
            cellText=[[name, value] for name, value in metrics],
            colLabels=["Metric", "Value"],
            colWidths=[0.78, 0.22],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.0, 1.9)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.7)
            if r == 0:
                cell.set_facecolor(HEADER)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor(ROW_ALT if r % 2 == 0 else "white")
                if c == 1:
                    cell.set_text_props(color=ACCENT, weight="bold")
                else:
                    cell.set_text_props(color=TEXT)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT_DIR / "quantitative_metrics_table_2col.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "quantitative_metrics_table_2col.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    Q, flags, time, _angles, _positions, _rotations, _camera_poses, _tcp_poses, traj_info = (
        compute_ik_trajectory(
            model,
            data,
            verbose=False,
            retries=DEFAULT_IK_RETRIES,
            max_waypoints=36,
        )
    )

    Q = np.asarray(Q, dtype=float)
    time = np.asarray(time, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    qdot = _gradient(Q, time)
    qddot = _gradient(qdot, time)
    jerk = _gradient(qddot, time)

    total_waypoints = len(Q)
    capture_valid = np.asarray(traj_info.get("capture_valid", flags), dtype=bool)
    capture_pos_err = np.asarray(traj_info.get("capture_pos_err", np.full(total_waypoints, np.nan)), dtype=float)
    orientation_err_deg = np.asarray(
        traj_info.get("target_rotation_err_deg", np.full(total_waypoints, np.nan)),
        dtype=float,
    )
    collision_free = np.asarray(traj_info.get("collision_free", np.ones(total_waypoints, dtype=bool)), dtype=bool)

    limits = joint_limits(model, mujoco)
    limit_margin = _joint_limit_margin(Q, limits)
    dq = np.diff(Q, axis=0)
    max_joint_step = float(np.max(np.abs(dq))) if len(dq) else 0.0

    metrics = [
        ("total waypoints", str(total_waypoints)),
        ("capture-valid waypoints", _fmt_count(int(np.count_nonzero(capture_valid)), total_waypoints)),
        ("max position error [mm]", _fmt_float(float(np.nanmax(capture_pos_err) * 1000.0), 3)),
        ("mean position error [mm]", _fmt_float(float(np.nanmean(capture_pos_err) * 1000.0), 3)),
        ("max orientation error [deg]", _fmt_float(float(np.nanmax(orientation_err_deg)), 3)),
        ("mean orientation error [deg]", _fmt_float(float(np.nanmean(orientation_err_deg)), 3)),
        ("collision-free frames", _fmt_count(int(np.count_nonzero(collision_free)), total_waypoints)),
        ("min joint limit margin [rad]", _fmt_float(float(np.nanmin(limit_margin)), 3)),
        ("max joint step [rad]", _fmt_float(max_joint_step, 3)),
        ("max joint velocity [rad/s]", _fmt_float(float(np.nanmax(np.abs(qdot))), 3)),
        ("max acceleration [rad/s²]", _fmt_float(float(np.nanmax(np.abs(qddot))), 3)),
        ("max jerk [rad/s³]", _fmt_float(float(np.nanmax(np.abs(jerk))), 3)),
    ]

    left = metrics[:6]
    right = metrics[6:]

    _write_csv(metrics)
    _draw_single_table(metrics)
    _draw_two_column_table(left, right)

    print(f"Wrote quantitative metrics to {OUT_DIR}")
    for name, value in metrics:
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
