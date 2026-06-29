"""MuJoCo scene overlay helpers for reference paths and tracking markers."""

from __future__ import annotations

import numpy as np


PAST_RGBA = np.array([0.0, 0.85, 1.0, 0.95], dtype=float)
FUTURE_RGBA = np.array([0.0, 0.85, 1.0, 0.22], dtype=float)
TCP_RGBA = np.array([1.0, 0.05, 0.03, 1.0], dtype=float)
TARGET_RGBA = np.array([0.05, 0.95, 0.20, 1.0], dtype=float)
GHOST_RGBA = np.array([0.95, 0.95, 1.0, 0.26], dtype=float)


def _geom_slot(scene):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def add_sphere(scene, mujoco, pos, radius: float, rgba) -> None:
    geom = _geom_slot(scene)
    if geom is None:
        return
    size = np.array([radius, radius, radius], dtype=float)
    mat = np.eye(3).reshape(-1)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        np.asarray(pos, dtype=float),
        mat,
        np.asarray(rgba, dtype=float),
    )


def add_line(scene, mujoco, p0, p1, rgba, width: float = 0.006) -> None:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
        return
    if np.linalg.norm(p1 - p0) < 1.0e-9:
        return
    geom = _geom_slot(scene)
    if geom is None:
        return
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        float(width),
        p0,
        p1,
    )
    geom.rgba[:] = np.asarray(rgba, dtype=float)


def add_polyline(scene, mujoco, points, rgba, width: float, max_segments: int = 180) -> None:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return
    if len(points) - 1 > max_segments:
        idx = np.linspace(0, len(points) - 1, max_segments + 1).round().astype(int)
        points = points[idx]
    for p0, p1 in zip(points[:-1], points[1:]):
        add_line(scene, mujoco, p0, p1, rgba=rgba, width=width)


def add_reference_overlay(
    scene,
    mujoco,
    ref_path=None,
    progress_index: int | None = None,
    tcp_pos=None,
    target_pos=None,
) -> None:
    if ref_path is not None:
        ref_path = np.asarray(ref_path, dtype=float)
        if len(ref_path) >= 2:
            idx = int(np.clip(progress_index if progress_index is not None else 0, 0, len(ref_path) - 1))
            add_polyline(scene, mujoco, ref_path[: idx + 1], PAST_RGBA, width=0.007)
            add_polyline(scene, mujoco, ref_path[idx:], FUTURE_RGBA, width=0.004)
    if tcp_pos is not None and np.isfinite(tcp_pos).all():
        add_sphere(scene, mujoco, tcp_pos, radius=0.018, rgba=TCP_RGBA)
    if target_pos is not None and np.isfinite(target_pos).all():
        add_sphere(scene, mujoco, target_pos, radius=0.018, rgba=TARGET_RGBA)


def add_ghost_skeleton(scene, mujoco, model, ghost_data, body_names: tuple[str, ...]) -> None:
    positions = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            positions.append(ghost_data.xpos[body_id].copy())
    if len(positions) < 2:
        return
    add_polyline(scene, mujoco, np.asarray(positions), GHOST_RGBA, width=0.010, max_segments=32)
    for pos in positions:
        add_sphere(scene, mujoco, pos, radius=0.012, rgba=GHOST_RGBA)
