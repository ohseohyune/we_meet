#!/usr/bin/env python3
"""Export presentation-ready joint motion plots from the current MuJoCo IK result."""

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

from mujoco_viewer import DEFAULT_IK_RETRIES, SCENE_XML, compute_ik_trajectory


OUT_DIR = PROJECT_ROOT / "outputs" / "presentation" / "joint_motion"
ACCENT = "#cf1b68"
MAROON = "#6b1730"
BLUE = "#1f77b4"
TEAL = "#008b8b"
ORANGE = "#dd8a16"
GREEN = "#2f9e44"
PURPLE = "#7b2cbf"
BROWN = "#8c564b"
GREY = "#767676"
GRID = "#d9d2c3"
JOINT_COLORS = [BLUE, TEAL, ACCENT, ORANGE, GREEN, PURPLE]


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


def _gradient(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, time, axis=0, edge_order=1)


def _segment_boundaries(time: np.ndarray, traj_info: dict) -> list[float]:
    segment_ids = np.asarray(traj_info.get("segment_id", []), dtype=int)
    if len(segment_ids) != len(time):
        return []
    idx = np.where(np.diff(segment_ids) != 0)[0] + 1
    return [float(time[i]) for i in idx]


def _draw_boundaries(ax, boundaries: list[float]) -> None:
    for boundary in boundaries:
        ax.axvline(boundary, color=GREY, linewidth=1.0, linestyle="--", alpha=0.55)


def _plot_joint_series(
    time: np.ndarray,
    values: np.ndarray,
    boundaries: list[float],
    title: str,
    ylabel: str,
    filename: str,
    ylim_pad_ratio: float = 0.12,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for j in range(values.shape[1]):
        ax.plot(time, values[:, j], linewidth=2.0, color=JOINT_COLORS[j], label=f"q{j + 1}")
    _draw_boundaries(ax, boundaries)
    _setup_axes(ax, title, "time [s]", ylabel)
    ax.legend(frameon=False, fontsize=9, ncol=6, loc="upper center")
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = max((high - low) * ylim_pad_ratio, 1e-3)
    ax.set_ylim(low - pad, high + pad)
    _save(fig, filename)


def _plot_combined(
    time: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    jerk: np.ndarray,
    boundaries: list[float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.1))
    series = [
        (q, r"joint angle $q(t)$", "angle [rad]"),
        (qdot, r"joint velocity $\dot{q}(t)$", "velocity [rad/s]"),
        (qddot, r"joint acceleration $\ddot{q}(t)$", "accel [rad/s²]"),
        (jerk, r"joint jerk $\dddot{q}(t)$", "jerk [rad/s³]"),
    ]
    for ax, (values, title, ylabel) in zip(axes.ravel(), series):
        for j in range(values.shape[1]):
            ax.plot(time, values[:, j], linewidth=1.7, color=JOINT_COLORS[j], label=f"q{j + 1}")
        _draw_boundaries(ax, boundaries)
        _setup_axes(ax, title, "time [s]", ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=6, loc="upper center")
    fig.tight_layout(pad=2.0)
    _save(fig, "joint_motion_summary_2x2")


def _write_metrics(
    time: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    jerk: np.ndarray,
    flags: np.ndarray,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "waypoints": len(time),
        "ik_success": int(np.sum(flags)),
        "ik_total": len(flags),
        "duration_s": float(time[-1] - time[0]) if len(time) else 0.0,
        "max_abs_joint_angle_rad": float(np.nanmax(np.abs(q))),
        "max_abs_joint_velocity_rad_s": float(np.nanmax(np.abs(qdot))),
        "max_abs_joint_accel_rad_s2": float(np.nanmax(np.abs(qddot))),
        "max_abs_joint_jerk_rad_s3": float(np.nanmax(np.abs(jerk))),
    }
    with open(OUT_DIR / "joint_motion_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    with open(OUT_DIR / "joint_motion_timeseries.csv", "w", newline="") as f:
        fieldnames = ["index", "time_s", "ik_success"]
        fieldnames += [f"q{j + 1}" for j in range(q.shape[1])]
        fieldnames += [f"qdot{j + 1}" for j in range(q.shape[1])]
        fieldnames += [f"qddot{j + 1}" for j in range(q.shape[1])]
        fieldnames += [f"jerk{j + 1}" for j in range(q.shape[1])]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(time)):
            row = {"index": i, "time_s": float(time[i]), "ik_success": int(flags[i])}
            row.update({f"q{j + 1}": float(q[i, j]) for j in range(q.shape[1])})
            row.update({f"qdot{j + 1}": float(qdot[i, j]) for j in range(q.shape[1])})
            row.update({f"qddot{j + 1}": float(qddot[i, j]) for j in range(q.shape[1])})
            row.update({f"jerk{j + 1}": float(jerk[i, j]) for j in range(q.shape[1])})
            writer.writerow(row)


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

    time = np.asarray(time, dtype=float)
    q = np.asarray(Q, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    qdot = _gradient(q, time)
    qddot = _gradient(qdot, time)
    jerk = _gradient(qddot, time)
    boundaries = _segment_boundaries(time, traj_info)

    _plot_joint_series(time, q, boundaries, r"joint angle $q_1 \sim q_6(t)$", "angle [rad]", "joint_angle")
    _plot_joint_series(time, qdot, boundaries, r"joint velocity $\dot{q}(t)$", "velocity [rad/s]", "joint_velocity")
    _plot_joint_series(time, qddot, boundaries, r"joint acceleration $\ddot{q}(t)$", "accel [rad/s²]", "joint_acceleration")
    _plot_joint_series(time, jerk, boundaries, r"joint jerk $\dddot{q}(t)$", "jerk [rad/s³]", "joint_jerk")
    _plot_combined(time, q, qdot, qddot, jerk, boundaries)
    _write_metrics(time, q, qdot, qddot, jerk, flags)

    print(f"Wrote joint motion plots to {OUT_DIR}")
    print(f"IK success: {int(np.sum(flags))}/{len(flags)}")
    print(f"Duration: {time[-1] - time[0]:.2f} s")
    print(f"Max |qdot|: {np.nanmax(np.abs(qdot)):.3f} rad/s")
    print(f"Max |qddot|: {np.nanmax(np.abs(qddot)):.3f} rad/s^2")
    print(f"Max |jerk|: {np.nanmax(np.abs(jerk)):.3f} rad/s^3")


if __name__ == "__main__":
    main()
