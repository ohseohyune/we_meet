#!/usr/bin/env python3
"""Create a presentation-ready frame visualization for the inspection setup."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np


PIPE_START_X = 0.56000
PIPE_LENGTH = 0.25
PIPE_END_X = PIPE_START_X + PIPE_LENGTH
PIPE_HEIGHT = 0.5700
PIPE_OD = 0.0605
SEAM_RADIUS = PIPE_OD / 2.0

FLANGE_CENTER = np.array([PIPE_END_X, 0.0, PIPE_HEIGHT])
TCP_CENTER = FLANGE_CENTER + np.array([-0.3000, 0.1200, 0.0000])
SEAM_SAMPLE = FLANGE_CENTER + np.array([0.0, SEAM_RADIUS * 0.78, SEAM_RADIUS * 0.62])


def equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins) * 1.18
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_arrow(
    ax,
    origin,
    vector,
    color,
    label,
    length_scale=1.0,
    lw=2.2,
    dashed=False,
    label_offset=(0.0, 0.0, 0.0),
):
    origin = np.asarray(origin, dtype=float)
    vector = np.asarray(vector, dtype=float) * length_scale
    linestyle = "--" if dashed else "-"
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        vector[0],
        vector[1],
        vector[2],
        color=color,
        linewidth=lw,
        arrow_length_ratio=0.18,
        linestyle=linestyle,
    )
    if label:
        tip = origin + vector + np.asarray(label_offset, dtype=float)
        ax.text(tip[0], tip[1], tip[2], label, color=color, fontsize=10, weight="bold")


def camera_frame(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return physical camera axes: local -Z points from camera to target."""
    look = target - position
    look /= np.linalg.norm(look) + 1e-12
    z_axis = -look
    y_ref = np.array([0.0, -0.62, 0.78])
    y_axis = y_ref - np.dot(y_ref, z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def draw_pipe(ax):
    x = np.linspace(PIPE_START_X, PIPE_END_X, 40)
    theta = np.linspace(0, 2 * np.pi, 48)
    xx, tt = np.meshgrid(x, theta)
    yy = SEAM_RADIUS * np.cos(tt)
    zz = PIPE_HEIGHT + SEAM_RADIUS * np.sin(tt)
    ax.plot_surface(xx, yy, zz, color="#9aa1a9", alpha=0.38, linewidth=0, shade=True)


def draw_flange(ax):
    radius = 0.120
    theta = np.linspace(0, 2 * np.pi, 120)
    y = radius * np.cos(theta)
    z = PIPE_HEIGHT + radius * np.sin(theta)
    x_front = np.full_like(theta, PIPE_END_X + 0.012)
    x_back = np.full_like(theta, PIPE_END_X - 0.012)
    ax.plot(x_front, y, z, color="#555b63", linewidth=2.0)
    ax.plot(x_back, y, z, color="#555b63", linewidth=1.5, alpha=0.7)
    for frac in np.linspace(0, 1, 18):
        r = radius * frac
        ax.plot(
            np.full_like(theta, PIPE_END_X),
            r * np.cos(theta),
            PIPE_HEIGHT + r * np.sin(theta),
            color="#747b84",
            linewidth=0.3,
            alpha=0.16,
        )


def draw_seam(ax):
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.full_like(theta, PIPE_END_X)
    y = SEAM_RADIUS * np.cos(theta)
    z = PIPE_HEIGHT + SEAM_RADIUS * np.sin(theta)
    ax.plot(x, y, z, color="#f05a28", linewidth=3.0, label="Pipe-Flange seam")


def draw_frame(ax, origin, R, prefix, scale=0.045, label_offset=(0.0, 0.0, 0.0), labels=True):
    x_label = f"{prefix} X" if labels else ""
    y_label = f"{prefix} Y" if labels else ""
    z_label = f"{prefix} Z" if labels else ""
    draw_arrow(ax, origin, R[:, 0], "#d62728", x_label, scale, label_offset=label_offset)
    draw_arrow(ax, origin, R[:, 1], "#2ca02c", y_label, scale, label_offset=label_offset)
    draw_arrow(ax, origin, R[:, 2], "#1f77b4", z_label, scale, label_offset=label_offset)


def main() -> None:
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    draw_pipe(ax)
    draw_flange(ax)
    draw_seam(ax)

    world_R = np.eye(3)
    flange_R = np.column_stack(
        [
            np.array([-1.0, 0.0, 0.0]),  # flange face normal toward robot
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
    )
    tcp_R = camera_frame(TCP_CENTER, SEAM_SAMPLE)

    draw_frame(ax, np.array([0.0, 0.0, 0.0]), world_R, "World", scale=0.085)
    draw_frame(ax, FLANGE_CENTER, flange_R, "PF", scale=0.060, labels=False)
    draw_frame(ax, TCP_CENTER, tcp_R, "TCP", scale=0.060, labels=False)

    optical = SEAM_SAMPLE - TCP_CENTER
    optical /= np.linalg.norm(optical) + 1e-12
    draw_arrow(
        ax,
        TCP_CENTER,
        optical,
        "#111111",
        "camera -Z / look",
        0.105,
        lw=2.0,
        dashed=True,
        label_offset=(0.0, -0.02, -0.015),
    )

    ax.scatter(*FLANGE_CENTER, color="#f05a28", s=42)
    ax.scatter(*TCP_CENTER, color="#111111", s=36)
    ax.scatter(*SEAM_SAMPLE, color="#f05a28", s=36)
    ax.text(*(FLANGE_CENTER + np.array([0.012, -0.068, -0.030])), "Pipe-Flange frame", color="#f05a28", fontsize=11, weight="bold")
    ax.text(*(TCP_CENTER + np.array([-0.040, 0.035, 0.035])), "TCP/Camera frame", color="#111111", fontsize=11, weight="bold")
    ax.text(*(SEAM_SAMPLE + np.array([0.018, 0.045, 0.025])), "seam target s(phi)", color="#f05a28", fontsize=10)

    ax.set_title("World/Base, Pipe-Flange, and TCP/Camera Frames", fontsize=16, pad=18)
    ax.set_xlabel("World X [m]")
    ax.set_ylabel("World Y [m]")
    ax.set_zlabel("World Z [m]")
    ax.view_init(elev=24, azim=-57)

    all_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [PIPE_START_X, -0.13, PIPE_HEIGHT - 0.13],
            [PIPE_END_X, 0.13, PIPE_HEIGHT + 0.13],
            TCP_CENTER,
            SEAM_SAMPLE,
        ]
    )
    equal_3d_axes(ax, all_points)
    ax.grid(True, alpha=0.25)

    notes = (
        "All waypoints are defined in the World/Base frame.  "
        "Pipe axis: +World X.  Pipe-Flange frame origin: flange face center.  "
        "TCP/Camera local -Z is aligned to the seam target.  "
        "Axis colors: X=red, Y=green, Z=blue."
    )
    fig.text(0.05, 0.03, notes, fontsize=10, color="#333333")

    png_path = out_dir / "frame_visualization.png"
    svg_path = out_dir / "frame_visualization.svg"
    fig.savefig(png_path, dpi=220)
    fig.savefig(svg_path)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
