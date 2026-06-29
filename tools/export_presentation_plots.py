#!/usr/bin/env python3
"""Export presentation-ready trajectory plots for the quintic segment slide."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reference.dream_math import quintic_spline
from trajectory.circle import (
    BOTTOM_FORBIDDEN_CENTER,
    BOTTOM_FORBIDDEN_HALF_ANGLE,
    SEGMENTS,
    segmented_circle_trajectory,
)
from trajectory.generator import (
    FLANGE_CENTER,
    SEAM_RADIUS,
    STANDOFF,
    TRAJECTORY_X_OFFSET,
)


OUT_DIR = PROJECT_ROOT / "outputs" / "presentation" / "quintic_spline"
ACCENT = "#cf1b68"
MAROON = "#6b1730"
BLUE = "#1f77b4"
TEAL = "#008b8b"
ORANGE = "#dd8a16"
GREEN = "#2f9e44"
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


def _plot_quintic_profiles() -> None:
    tau = np.linspace(0.0, 1.0, 401)
    values = np.array(
        [quintic_spline(t, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0) for t in tau]
    )
    profiles = [
        ("quintic_s", r"$s(t)$", "normalized position", values[:, 0], ACCENT),
        ("quintic_sdot", r"$\dot{s}(t)$", "normalized velocity", values[:, 1], TEAL),
        ("quintic_sddot", r"$\ddot{s}(t)$", "normalized acceleration", values[:, 2], ORANGE),
    ]

    for filename, title, ylabel, y, color in profiles:
        fig, ax = plt.subplots(figsize=(4.0, 3.4))
        ax.plot(tau, y, color=color, linewidth=3.0)
        ax.scatter([tau[0], tau[-1]], [y[0], y[-1]], s=42, color=MAROON, zorder=3)
        _setup_axes(ax, title, "normalized time", ylabel)
        ax.set_xlim(0.0, 1.0)
        ypad = max((np.max(y) - np.min(y)) * 0.12, 0.05)
        ax.set_ylim(np.min(y) - ypad, np.max(y) + ypad)
        _save(fig, filename)


def _plot_phi_and_camera_positions() -> None:
    segment_duration = 9.0
    trajectory_center = FLANGE_CENTER + np.array([TRAJECTORY_X_OFFSET, 0.0, 0.0])
    traj = segmented_circle_trajectory(
        center=trajectory_center,
        radius=STANDOFF,
        segment_duration=segment_duration,
        dt=0.02,
        orientation_target=FLANGE_CENTER,
        target_radius=SEAM_RADIUS,
    )
    time = traj["time"]
    phi_deg = np.rad2deg(traj["angles"])
    pos = traj["positions"]
    segment_id = traj["segment_id"]

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    for sid in np.unique(segment_id):
        mask = segment_id == sid
        ax.plot(time[mask], phi_deg[mask], linewidth=3.0, label=f"seg {sid}")
    for boundary in np.arange(segment_duration, segment_duration * len(SEGMENTS), segment_duration):
        ax.axvline(boundary, color=GREY, linewidth=1.0, linestyle="--", alpha=0.65)
    _setup_axes(ax, r"$\phi(t)$", "time [s]", "inspection angle [deg]")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="best")
    _save(fig, "phi_segments")

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    labels = ("camera x", "camera y", "camera z")
    colors = (BLUE, TEAL, ACCENT)
    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(time, pos[:, idx], linewidth=2.6, color=color, label=label)
    for boundary in np.arange(segment_duration, segment_duration * len(SEGMENTS), segment_duration):
        ax.axvline(boundary, color=GREY, linewidth=1.0, linestyle="--", alpha=0.5)
    _setup_axes(ax, "camera position x / y / z", "time [s]", "position [m]")
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper center")
    _save(fig, "camera_position_xyz")

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    y_offset = pos[:, 1] - trajectory_center[1]
    z_offset = pos[:, 2] - trajectory_center[2]
    ax.plot(time, y_offset, linewidth=2.8, color=TEAL, label=r"$y-y_c=R\cos\phi(t)$")
    ax.plot(time, z_offset, linewidth=2.8, color=ACCENT, label=r"$z-z_c=R\sin\phi(t)$")
    for boundary in np.arange(segment_duration, segment_duration * len(SEGMENTS), segment_duration):
        ax.axvline(boundary, color=GREY, linewidth=1.0, linestyle="--", alpha=0.5)
    ax.axhline(0.0, color="#999999", linewidth=0.9, alpha=0.65)
    _setup_axes(ax, "camera y/z offset from flange center", "time [s]", "offset [m]")
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center")
    _save(fig, "camera_yz_cos_sin_offset")


def _plot_segment_diagram() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    seam_y = SEAM_RADIUS * np.cos(theta)
    seam_z = SEAM_RADIUS * np.sin(theta)
    cam_y = STANDOFF * np.cos(theta)
    cam_z = STANDOFF * np.sin(theta)
    ax.plot(seam_y, seam_z, color=MAROON, linewidth=2.5, label="seam circle")
    ax.plot(cam_y, cam_z, color=ACCENT, linewidth=2.5, linestyle="--", label="reference trajectory")

    forbidden = np.linspace(
        BOTTOM_FORBIDDEN_CENTER - BOTTOM_FORBIDDEN_HALF_ANGLE,
        BOTTOM_FORBIDDEN_CENTER + BOTTOM_FORBIDDEN_HALF_ANGLE,
        100,
    )
    ax.fill_between(
        STANDOFF * np.cos(forbidden),
        STANDOFF * np.sin(forbidden),
        -STANDOFF * 1.2,
        color="#f4d7df",
        alpha=0.55,
        label="support-bar exclusion",
    )

    segment_colors = [BLUE, TEAL, ORANGE, GREEN]
    for idx, (segment, color) in enumerate(zip(SEGMENTS, segment_colors), start=1):
        phis = np.linspace(segment["phi0"], segment["phif"], 80)
        y = STANDOFF * np.cos(phis)
        z = STANDOFF * np.sin(phis)
        ax.plot(y, z, color=color, linewidth=4.0, alpha=0.9)
        mid = len(phis) // 2
        ax.annotate(
            "",
            xy=(y[mid + 1], z[mid + 1]),
            xytext=(y[mid - 2], z[mid - 2]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.2},
        )
        ax.text(y[mid], z[mid], f"S{idx}", color=color, fontsize=11, fontweight="bold")

    ax.scatter([0.0], [0.0], s=38, color="#111111", zorder=4)
    ax.text(0.004, 0.004, "flange center", fontsize=9, color="#111111")
    ax.set_aspect("equal", adjustable="box")
    _setup_axes(ax, "segmented seam circle", "world y [m]", "world z offset [m]")
    margin = STANDOFF * 1.25
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _save(fig, "seam_segment_diagram")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )
    _plot_quintic_profiles()
    _plot_phi_and_camera_positions()
    _plot_segment_diagram()
    print(f"Wrote presentation plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
