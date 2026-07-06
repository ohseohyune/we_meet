#!/usr/bin/env python3
"""Export synchronized D405 images and robot/camera pose metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.ik_solver import evaluate_look_at_trajectory, set_arm_qpos, site_pose, solve_trajectory
from mujoco_viewer import (
    CAPTURE_MAX_LOOK_DEG,
    CAPTURE_MAX_POS_ERR,
    CAMERA_SITE_NAME,
    D405_DEPTH_HEIGHT,
    D405_DEPTH_WIDTH,
    D405_MAX_Z,
    D405_MIN_Z,
    D405_VERTICAL_FOV_DEG,
    EE_LOOK_AXIS_COL,
    EE_LOOK_AXIS_SIGN,
    FLANGE_CENTER,
    NDOF,
    RIGHT_BIASED_READY,
    RIGHT_POSTURE_BIAS,
    SCENE_XML,
    SEAM_TARGET_RADIUS,
    TRAJECTORY_CENTER,
    TRAJECTORY_RADIUS,
    camera_poses_to_tcp_poses,
    generate_segmented_reference,
    mask_d405_depth,
    right_posture_weights,
    save_depth_png,
    set_d405_depth_rendering,
)
from trajectory.circle import (
    DEFAULT_MULTI_RING_SPECS,
    estimate_frames_for_overlap,
    estimate_multi_ring_frames,
)
from trajectory.generator import rot_to_quat


def transform_from_pos_rot(pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    return T


def flatten_transform(prefix: str, T: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{r}{c}": float(T[r, c]) for r in range(4) for c in range(4)}


def first_available_site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: tuple[str, ...],
) -> np.ndarray:
    for name in names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return site_pose(model, data, mujoco, name)
    raise ValueError(f"None of these MuJoCo sites exist: {names}")


def pinhole_intrinsics(width: int, height: int, fovy_deg: float) -> dict[str, float]:
    fovy = np.deg2rad(fovy_deg)
    fy = 0.5 * height / np.tan(0.5 * fovy)
    fx = fy
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float((width - 1) * 0.5),
        "cy": float((height - 1) * 0.5),
        "fovy_deg": float(fovy_deg),
        "fovx_deg": float(2.0 * np.rad2deg(np.arctan((width / height) * np.tan(0.5 * fovy)))),
        "depth_min_m": float(D405_MIN_Z),
        "depth_max_m": float(D405_MAX_Z),
    }


def prepare_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("rgb", "depth_png", "depth_meters"):
        path = out_dir / subdir
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    for filename in ("metadata.csv", "camera_intrinsics.json", "README.txt"):
        path = out_dir / filename
        if path.exists():
            path.unlink()


def save_rgb_png(rgb: np.ndarray, path: Path) -> None:
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    plt.imsave(path, rgb)


def black_rgb_frame() -> np.ndarray:
    return np.zeros((D405_DEPTH_HEIGHT, D405_DEPTH_WIDTH, 3), dtype=np.uint8)


def empty_depth_frame() -> np.ndarray:
    return np.full((D405_DEPTH_HEIGHT, D405_DEPTH_WIDTH), np.nan, dtype=np.float32)


def render_frame_pair(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    rgb_renderer: mujoco.Renderer,
    depth_renderer: mujoco.Renderer,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_renderer.update_scene(data, camera=camera_id)
    rgb = rgb_renderer.render()

    depth_renderer.update_scene(data, camera=camera_id)
    depth_m = depth_renderer.render().astype(np.float32)
    return rgb, depth_m


def solve_dataset_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tcp_poses: list[np.ndarray],
    look_targets: np.ndarray,
    traj_info: dict,
    multi_ring: bool,
    retries: int,
    rng_seed: int,
) -> tuple[np.ndarray, list[bool]]:
    """Solve IK, resetting the seed at each independent multi-ring pass."""
    posture_weights = right_posture_weights(traj_info, len(tcp_poses))
    if not multi_ring:
        return solve_trajectory(
            model,
            data,
            mujoco,
            tcp_poses,
            look_target=look_targets,
            q_start=RIGHT_BIASED_READY.copy(),
            retries=retries,
            rng_seed=rng_seed,
            verbose=False,
            axis_col=EE_LOOK_AXIS_COL,
            axis_sign=EE_LOOK_AXIS_SIGN,
            posture_bias=RIGHT_POSTURE_BIAS,
            posture_weights=posture_weights,
            max_capture_look_deg=CAPTURE_MAX_LOOK_DEG,
            max_capture_pos_err=CAPTURE_MAX_POS_ERR,
            site_name=CAMERA_SITE_NAME,
        )

    ring_ids = np.asarray(traj_info["ring_id"], dtype=int)
    Q = np.zeros((len(tcp_poses), NDOF))
    flags: list[bool] = [False] * len(tcp_poses)

    for ring_id in np.unique(ring_ids):
        idx = np.flatnonzero(ring_ids == ring_id)
        tcp_ring = [tcp_poses[i] for i in idx]
        targets_ring = look_targets[idx]
        posture_weights_ring = posture_weights[idx]
        Q_ring, flags_ring = solve_trajectory(
            model,
            data,
            mujoco,
            tcp_ring,
            look_target=targets_ring,
            q_start=RIGHT_BIASED_READY.copy(),
            retries=retries,
            rng_seed=rng_seed + int(ring_id),
            verbose=False,
            axis_col=EE_LOOK_AXIS_COL,
            axis_sign=EE_LOOK_AXIS_SIGN,
            posture_bias=RIGHT_POSTURE_BIAS,
            posture_weights=posture_weights_ring,
            max_capture_look_deg=CAPTURE_MAX_LOOK_DEG,
            max_capture_pos_err=CAPTURE_MAX_POS_ERR,
            site_name=CAMERA_SITE_NAME,
        )
        Q[idx] = Q_ring
        for local_i, ok in enumerate(flags_ring):
            flags[int(idx[local_i])] = ok

    return Q, flags


def export_dataset(
    scene_path: str,
    out_dir: Path,
    n_frames: int,
    start_index: int,
    camera_name: str,
    retries: int,
    rng_seed: int,
    multi_ring: bool,
    overlap: float | None,
    min_frames_per_ring: int,
    insert_separators: bool,
) -> None:
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"MuJoCo camera '{camera_name}' not found.")

    frame_plan = {
        "mode": "multi_ring" if multi_ring else "single_ring",
        "requested_frames": int(n_frames),
        "overlap": None if overlap is None else float(overlap),
        "min_frames_per_ring": int(min_frames_per_ring),
    }
    if overlap is not None:
        if multi_ring:
            n_frames, per_ring = estimate_multi_ring_frames(
                ring_specs=DEFAULT_MULTI_RING_SPECS,
                target_radius=SEAM_TARGET_RADIUS,
                fovy_deg=D405_VERTICAL_FOV_DEG,
                width=D405_DEPTH_WIDTH,
                height=D405_DEPTH_HEIGHT,
                overlap=overlap,
                min_frames_per_ring=min_frames_per_ring,
            )
            frame_plan["auto_frames_per_ring"] = per_ring
        else:
            n_frames = estimate_frames_for_overlap(
                camera_x_offset=float(TRAJECTORY_CENTER[0] - FLANGE_CENTER[0]),
                camera_radius=float(TRAJECTORY_RADIUS),
                target_radius=SEAM_TARGET_RADIUS,
                fovy_deg=D405_VERTICAL_FOV_DEG,
                width=D405_DEPTH_WIDTH,
                height=D405_DEPTH_HEIGHT,
                overlap=overlap,
                min_frames=min_frames_per_ring,
            )
        frame_plan["computed_frames"] = int(n_frames)

    ref = generate_segmented_reference(
        max_waypoints=n_frames,
        return_targets=True,
        multi_ring=multi_ring,
        return_info=True,
    )
    time_values, angles, camera_positions, _, camera_poses, look_targets, traj_info = ref
    tcp_poses = camera_poses_to_tcp_poses(camera_poses)
    separator_count = 0
    if insert_separators:
        segment_ids_preview = traj_info.get("segment_id", np.zeros(len(time_values), dtype=int))
        ring_ids_preview = traj_info.get("ring_id", np.ones(len(time_values), dtype=int))
        for i in range(1, len(time_values)):
            if (
                int(segment_ids_preview[i]) != int(segment_ids_preview[i - 1])
                or int(ring_ids_preview[i]) != int(ring_ids_preview[i - 1])
            ):
                separator_count += 1

    print(f"[DATASET] Solving IK for {n_frames} synchronized capture poses...")
    Q, flags = solve_dataset_ik(
        model=model,
        data=data,
        tcp_poses=tcp_poses,
        look_targets=look_targets,
        traj_info=traj_info,
        multi_ring=multi_ring,
        retries=retries,
        rng_seed=rng_seed,
    )

    capture_metrics = evaluate_look_at_trajectory(
        model,
        data,
        mujoco,
        Q,
        tcp_poses,
        look_targets,
        axis_col=EE_LOOK_AXIS_COL,
        axis_sign=EE_LOOK_AXIS_SIGN,
        site_name=CAMERA_SITE_NAME,
        max_pos_err=CAPTURE_MAX_POS_ERR,
        max_look_deg=CAPTURE_MAX_LOOK_DEG,
    )
    capture_valid = capture_metrics["capture_valid"]
    flags = [bool(ok) for ok in capture_valid]

    n_ok = int(np.count_nonzero(capture_valid))
    invalid_capture_count = int(len(flags) - n_ok)
    if invalid_capture_count:
        failed = [i for i, ok in enumerate(capture_valid) if not ok]
        print(f"[WARN] Look-at/position gate failed for frames: {failed}")
    print(
        f"[DATASET] Capture-valid frames: {n_ok}/{len(flags)} "
        f"(look <= {CAPTURE_MAX_LOOK_DEG:.1f}deg, pos <= {CAPTURE_MAX_POS_ERR*1000:.1f}mm)"
    )

    prepare_output_dir(out_dir)

    set_d405_depth_rendering(model, camera_id)
    intrinsics = pinhole_intrinsics(D405_DEPTH_WIDTH, D405_DEPTH_HEIGHT, D405_VERTICAL_FOV_DEG)
    with open(out_dir / "camera_intrinsics.json", "w") as f:
        json.dump(
            {
                "camera_name": camera_name,
                "model": "Intel RealSense D405 style MuJoCo fixed camera",
                "intrinsics": intrinsics,
                "notes": [
                    "Depth arrays are metric meters from mujoco.Renderer.enable_depth_rendering().",
                    "Pixels outside 0.07 m ~ 0.50 m are masked in depth_png visualizations only.",
                    "metadata.csv stores the actual MuJoCo render camera pose from data.cam_xpos/cam_xmat.",
                ],
                "frame_range": {
                    "start": int(start_index),
                    "end": int(start_index + n_frames + separator_count - 1),
                    "count": int(n_frames + separator_count),
                    "requested_capture_count": int(n_frames),
                    "valid_capture_count": int(n_ok),
                    "invalid_capture_count": int(invalid_capture_count),
                    "separator_count": int(separator_count),
                },
                "capture_gate": {
                    "max_look_deg": float(CAPTURE_MAX_LOOK_DEG),
                    "max_position_error_m": float(CAPTURE_MAX_POS_ERR),
                    "invalid_frames_are_black": True,
                },
                "frame_plan": frame_plan,
                "insert_separators": bool(insert_separators),
                "ring_specs": [dict(ring) for ring in DEFAULT_MULTI_RING_SPECS] if multi_ring else [],
            },
            f,
            indent=2,
        )

    rgb_renderer = mujoco.Renderer(model, height=D405_DEPTH_HEIGHT, width=D405_DEPTH_WIDTH)
    depth_renderer = mujoco.Renderer(model, height=D405_DEPTH_HEIGHT, width=D405_DEPTH_WIDTH)
    depth_renderer.enable_depth_rendering()

    metadata_path = out_dir / "metadata.csv"
    base_fields = [
        "capture_index",
        "frame",
        "is_separator",
        "separator_reason",
        "ring_id",
        "ring_name",
        "ring_radius",
        "ring_x_offset",
        "segment_id",
        "segment_name",
        "time_s",
        "segment_angle_rad",
        "segment_angle_deg",
        "ik_success",
        "capture_valid",
        "invalid_reason",
        "capture_look_error_deg",
        "capture_position_error_mm",
        "rgb_path",
        "depth_png_path",
        "depth_npy_path",
        "valid_depth_pixels",
        "depth_min_valid_m",
        "depth_max_valid_m",
        "desired_camera_x",
        "desired_camera_y",
        "desired_camera_z",
        "seam_target_x",
        "seam_target_y",
        "seam_target_z",
        "render_camera_x",
        "render_camera_y",
        "render_camera_z",
        "render_camera_qw",
        "render_camera_qx",
        "render_camera_qy",
        "render_camera_qz",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "tcp_qw",
        "tcp_qx",
        "tcp_qy",
        "tcp_qz",
        "optical_site_x",
        "optical_site_y",
        "optical_site_z",
        "optical_site_qw",
        "optical_site_qx",
        "optical_site_qy",
        "optical_site_qz",
    ]
    q_fields = [f"q{i}" for i in range(1, NDOF + 1)]
    T_fields = (
        list(flatten_transform("T_world_render_camera", np.eye(4)).keys())
        + list(flatten_transform("T_world_tcp", np.eye(4)).keys())
        + list(flatten_transform("T_world_optical_site", np.eye(4)).keys())
    )

    print(f"[DATASET] Rendering RGB + D405 depth frames into {out_dir}/")
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + q_fields + T_fields)
        writer.writeheader()

        ring_ids = traj_info.get("ring_id", np.ones(len(time_values), dtype=int))
        ring_names = traj_info.get("ring_name", np.asarray(["single_ring"] * len(time_values)))
        ring_radii = traj_info.get("ring_radius", np.full(len(time_values), TRAJECTORY_RADIUS))
        ring_x_offsets = traj_info.get(
            "ring_x_offset",
            np.full(len(time_values), float(TRAJECTORY_CENTER[0] - FLANGE_CENTER[0])),
        )
        segment_ids = traj_info.get("segment_id", np.zeros(len(time_values), dtype=int))
        segment_names = traj_info.get("segment_name", np.asarray([""] * len(time_values)))

        output_i = 0

        def write_separator(frame_number: int, reason: str, prev_i: int) -> None:
            rgb_rel = Path("rgb") / f"frame_{frame_number:03d}.png"
            depth_png_rel = Path("depth_png") / f"frame_{frame_number:03d}.png"
            depth_npy_rel = Path("depth_meters") / f"frame_{frame_number:03d}.npy"
            depth_m = empty_depth_frame()

            save_rgb_png(black_rgb_frame(), out_dir / rgb_rel)
            save_rgb_png(black_rgb_frame(), out_dir / depth_png_rel)
            np.save(out_dir / depth_npy_rel, depth_m)

            T_nan = np.full((4, 4), np.nan)
            row = {
                "capture_index": "",
                "frame": frame_number,
                "is_separator": 1,
                "separator_reason": reason,
                "ring_id": int(ring_ids[prev_i]) if prev_i >= 0 else "",
                "ring_name": str(ring_names[prev_i]) if prev_i >= 0 else "",
                "ring_radius": float(ring_radii[prev_i]) if prev_i >= 0 else np.nan,
                "ring_x_offset": float(ring_x_offsets[prev_i]) if prev_i >= 0 else np.nan,
                "segment_id": int(segment_ids[prev_i]) if prev_i >= 0 else "",
                "segment_name": "separator",
                "time_s": "",
                "segment_angle_rad": "",
                "segment_angle_deg": "",
                "ik_success": "",
                "capture_valid": "",
                "invalid_reason": "",
                "capture_look_error_deg": "",
                "capture_position_error_mm": "",
                "rgb_path": str(rgb_rel),
                "depth_png_path": str(depth_png_rel),
                "depth_npy_path": str(depth_npy_rel),
                "valid_depth_pixels": 0,
                "depth_min_valid_m": np.nan,
                "depth_max_valid_m": np.nan,
                "desired_camera_x": np.nan,
                "desired_camera_y": np.nan,
                "desired_camera_z": np.nan,
                "seam_target_x": np.nan,
                "seam_target_y": np.nan,
                "seam_target_z": np.nan,
                "render_camera_x": np.nan,
                "render_camera_y": np.nan,
                "render_camera_z": np.nan,
                "render_camera_qw": np.nan,
                "render_camera_qx": np.nan,
                "render_camera_qy": np.nan,
                "render_camera_qz": np.nan,
                "tcp_x": np.nan,
                "tcp_y": np.nan,
                "tcp_z": np.nan,
                "tcp_qw": np.nan,
                "tcp_qx": np.nan,
                "tcp_qy": np.nan,
                "tcp_qz": np.nan,
                "optical_site_x": np.nan,
                "optical_site_y": np.nan,
                "optical_site_z": np.nan,
                "optical_site_qw": np.nan,
                "optical_site_qx": np.nan,
                "optical_site_qy": np.nan,
                "optical_site_qz": np.nan,
            }
            row.update({f"q{j + 1}": np.nan for j in range(NDOF)})
            row.update(flatten_transform("T_world_render_camera", T_nan))
            row.update(flatten_transform("T_world_tcp", T_nan))
            row.update(flatten_transform("T_world_optical_site", T_nan))
            writer.writerow(row)

        def write_invalid_capture(frame_number: int, reason: str, i: int, q: np.ndarray) -> None:
            rgb_rel = Path("rgb") / f"frame_{frame_number:03d}.png"
            depth_png_rel = Path("depth_png") / f"frame_{frame_number:03d}.png"
            depth_npy_rel = Path("depth_meters") / f"frame_{frame_number:03d}.npy"
            depth_m = empty_depth_frame()

            save_rgb_png(black_rgb_frame(), out_dir / rgb_rel)
            save_rgb_png(black_rgb_frame(), out_dir / depth_png_rel)
            np.save(out_dir / depth_npy_rel, depth_m)

            set_arm_qpos(model, data, mujoco, q)
            T_tcp = first_available_site_pose(model, data, ("tcp", "ee_site"))
            T_optical_site = first_available_site_pose(
                model,
                data,
                ("camera_optical_center", "ee_site"),
            )
            T_render_camera = transform_from_pos_rot(
                data.cam_xpos[camera_id].copy(),
                data.cam_xmat[camera_id].reshape(3, 3).copy(),
            )
            q_render = rot_to_quat(T_render_camera[:3, :3])
            q_tcp = rot_to_quat(T_tcp[:3, :3])
            q_optical = rot_to_quat(T_optical_site[:3, :3])

            row = {
                "capture_index": i,
                "frame": frame_number,
                "is_separator": 1,
                "separator_reason": reason,
                "ring_id": int(ring_ids[i]),
                "ring_name": str(ring_names[i]),
                "ring_radius": float(ring_radii[i]),
                "ring_x_offset": float(ring_x_offsets[i]),
                "segment_id": int(segment_ids[i]),
                "segment_name": str(segment_names[i]),
                "time_s": float(time_values[i]),
                "segment_angle_rad": float(angles[i]),
                "segment_angle_deg": float(np.rad2deg(angles[i])),
                "ik_success": int(flags[i]),
                "capture_valid": 0,
                "invalid_reason": reason,
                "capture_look_error_deg": float(capture_metrics["look_deg"][i]),
                "capture_position_error_mm": float(capture_metrics["pos_err"][i]) * 1000.0,
                "rgb_path": str(rgb_rel),
                "depth_png_path": str(depth_png_rel),
                "depth_npy_path": str(depth_npy_rel),
                "valid_depth_pixels": 0,
                "depth_min_valid_m": np.nan,
                "depth_max_valid_m": np.nan,
                "desired_camera_x": float(camera_poses[i][0, 3]),
                "desired_camera_y": float(camera_poses[i][1, 3]),
                "desired_camera_z": float(camera_poses[i][2, 3]),
                "seam_target_x": float(look_targets[i][0]),
                "seam_target_y": float(look_targets[i][1]),
                "seam_target_z": float(look_targets[i][2]),
                "render_camera_x": float(T_render_camera[0, 3]),
                "render_camera_y": float(T_render_camera[1, 3]),
                "render_camera_z": float(T_render_camera[2, 3]),
                "render_camera_qw": float(q_render[0]),
                "render_camera_qx": float(q_render[1]),
                "render_camera_qy": float(q_render[2]),
                "render_camera_qz": float(q_render[3]),
                "tcp_x": float(T_tcp[0, 3]),
                "tcp_y": float(T_tcp[1, 3]),
                "tcp_z": float(T_tcp[2, 3]),
                "tcp_qw": float(q_tcp[0]),
                "tcp_qx": float(q_tcp[1]),
                "tcp_qy": float(q_tcp[2]),
                "tcp_qz": float(q_tcp[3]),
                "optical_site_x": float(T_optical_site[0, 3]),
                "optical_site_y": float(T_optical_site[1, 3]),
                "optical_site_z": float(T_optical_site[2, 3]),
                "optical_site_qw": float(q_optical[0]),
                "optical_site_qx": float(q_optical[1]),
                "optical_site_qy": float(q_optical[2]),
                "optical_site_qz": float(q_optical[3]),
            }
            row.update({f"q{j + 1}": float(q[j]) for j in range(NDOF)})
            row.update(flatten_transform("T_world_render_camera", T_render_camera))
            row.update(flatten_transform("T_world_tcp", T_tcp))
            row.update(flatten_transform("T_world_optical_site", T_optical_site))
            writer.writerow(row)

        for i, (q, ok, t, angle, T_cam_des, seam_target) in enumerate(
            zip(Q, flags, time_values, angles, camera_poses, look_targets)
        ):
            if insert_separators and i > 0:
                reasons = []
                if int(segment_ids[i]) != int(segment_ids[i - 1]):
                    reasons.append(f"segment_{int(segment_ids[i - 1])}_to_{int(segment_ids[i])}")
                if int(ring_ids[i]) != int(ring_ids[i - 1]):
                    reasons.append(f"ring_{int(ring_ids[i - 1])}_to_{int(ring_ids[i])}")
                if reasons:
                    write_separator(start_index + output_i, "+".join(reasons), i - 1)
                    output_i += 1

            frame_number = start_index + output_i
            if not bool(capture_valid[i]):
                write_invalid_capture(frame_number, "invalid_capture:look_or_position", i, q)
                output_i += 1
                continue

            set_arm_qpos(model, data, mujoco, q)

            rgb, depth_m = render_frame_pair(
                model, data, camera_id, rgb_renderer, depth_renderer
            )

            rgb_rel = Path("rgb") / f"frame_{frame_number:03d}.png"
            depth_png_rel = Path("depth_png") / f"frame_{frame_number:03d}.png"
            depth_npy_rel = Path("depth_meters") / f"frame_{frame_number:03d}.npy"

            save_rgb_png(rgb, out_dir / rgb_rel)
            np.save(out_dir / depth_npy_rel, depth_m)
            save_depth_png(depth_m, out_dir / depth_png_rel)

            depth_valid = mask_d405_depth(depth_m).compressed()
            if depth_valid.size:
                depth_min = float(depth_valid.min())
                depth_max = float(depth_valid.max())
            else:
                depth_min = float("nan")
                depth_max = float("nan")

            T_tcp = first_available_site_pose(model, data, ("tcp", "ee_site"))
            T_optical_site = first_available_site_pose(
                model,
                data,
                ("camera_optical_center", "ee_site"),
            )
            T_render_camera = transform_from_pos_rot(
                data.cam_xpos[camera_id].copy(),
                data.cam_xmat[camera_id].reshape(3, 3).copy(),
            )

            q_render = rot_to_quat(T_render_camera[:3, :3])
            q_tcp = rot_to_quat(T_tcp[:3, :3])
            q_optical = rot_to_quat(T_optical_site[:3, :3])

            row = {
                "capture_index": i,
                "frame": frame_number,
                "is_separator": 0,
                "separator_reason": "",
                "ring_id": int(ring_ids[i]),
                "ring_name": str(ring_names[i]),
                "ring_radius": float(ring_radii[i]),
                "ring_x_offset": float(ring_x_offsets[i]),
                "segment_id": int(segment_ids[i]),
                "segment_name": str(segment_names[i]),
                "time_s": float(t),
                "segment_angle_rad": float(angle),
                "segment_angle_deg": float(np.rad2deg(angle)),
                "ik_success": int(ok),
                "capture_valid": int(capture_valid[i]),
                "invalid_reason": "",
                "capture_look_error_deg": float(capture_metrics["look_deg"][i]),
                "capture_position_error_mm": float(capture_metrics["pos_err"][i]) * 1000.0,
                "rgb_path": str(rgb_rel),
                "depth_png_path": str(depth_png_rel),
                "depth_npy_path": str(depth_npy_rel),
                "valid_depth_pixels": int(depth_valid.size),
                "depth_min_valid_m": depth_min,
                "depth_max_valid_m": depth_max,
                "desired_camera_x": float(T_cam_des[0, 3]),
                "desired_camera_y": float(T_cam_des[1, 3]),
                "desired_camera_z": float(T_cam_des[2, 3]),
                "seam_target_x": float(seam_target[0]),
                "seam_target_y": float(seam_target[1]),
                "seam_target_z": float(seam_target[2]),
                "render_camera_x": float(T_render_camera[0, 3]),
                "render_camera_y": float(T_render_camera[1, 3]),
                "render_camera_z": float(T_render_camera[2, 3]),
                "render_camera_qw": float(q_render[0]),
                "render_camera_qx": float(q_render[1]),
                "render_camera_qy": float(q_render[2]),
                "render_camera_qz": float(q_render[3]),
                "tcp_x": float(T_tcp[0, 3]),
                "tcp_y": float(T_tcp[1, 3]),
                "tcp_z": float(T_tcp[2, 3]),
                "tcp_qw": float(q_tcp[0]),
                "tcp_qx": float(q_tcp[1]),
                "tcp_qy": float(q_tcp[2]),
                "tcp_qz": float(q_tcp[3]),
                "optical_site_x": float(T_optical_site[0, 3]),
                "optical_site_y": float(T_optical_site[1, 3]),
                "optical_site_z": float(T_optical_site[2, 3]),
                "optical_site_qw": float(q_optical[0]),
                "optical_site_qx": float(q_optical[1]),
                "optical_site_qy": float(q_optical[2]),
                "optical_site_qz": float(q_optical[3]),
            }
            row.update({f"q{j + 1}": float(q[j]) for j in range(NDOF)})
            row.update(flatten_transform("T_world_render_camera", T_render_camera))
            row.update(flatten_transform("T_world_tcp", T_tcp))
            row.update(flatten_transform("T_world_optical_site", T_optical_site))
            writer.writerow(row)
            output_i += 1

    rgb_renderer.close()
    depth_renderer.close()

    with open(out_dir / "README.txt", "w") as f:
        f.write(
            "Inspection dataset export\n"
            "=========================\n\n"
            f"Frame range: frame_{start_index:03d} ~ frame_{start_index + n_frames + separator_count - 1:03d}\n"
            f"Requested capture poses: {n_frames}\n"
            f"Valid capture count: {n_ok}\n"
            f"Invalid black capture frames: {invalid_capture_count}\n\n"
            f"Separator frames: {separator_count}\n"
            f"Trajectory mode: {'multi-ring' if multi_ring else 'single-ring'}\n"
            f"Overlap request: {overlap if overlap is not None else 'manual frame count'}\n\n"
            "Each frame index has synchronized files:\n"
            "  rgb/frame_XXX.png          RGB render from d405_camera\n"
            "  depth_png/frame_XXX.png    depth visualization, terrain colormap\n"
            "  depth_meters/frame_XXX.npy raw metric depth array in meters\n"
            "  metadata.csv               joints, TCP pose, optical site pose, render camera pose\n\n"
            "Frames with capture_valid=0 are black invalid-capture frames and should be\n"
            "excluded from stitching/reconstruction.\n\n"
            "Use T_world_render_camera for image registration/stitching because it is the\n"
            "actual MuJoCo fixed-camera pose used by the renderer.\n"
        )

    with open(out_dir / "README.md", "w") as f:
        f.write(
            "# Pipe Flange Inspection Dataset\n\n"
            "MuJoCo 6-DOF DH robot + D405-style camera inspection dataset입니다.\n\n"
            f"- Frame range: `frame_{start_index:03d}` ~ `frame_{start_index + n_frames + separator_count - 1:03d}`\n"
            f"- Requested capture poses: `{n_frames}`\n"
            f"- Valid capture count: `{n_ok}`\n"
            f"- Invalid black capture frames: `{invalid_capture_count}`\n"
            f"- Separator frames: `{separator_count}`\n"
            f"- Trajectory mode: `{'multi-ring' if multi_ring else 'single-ring'}`\n"
            f"- Overlap request: `{overlap if overlap is not None else 'manual frame count'}`\n"
            "- RGB resolution: `1280 x 800`\n"
            "- Depth unit: meter\n"
            "- D405 depth range: `0.07 m ~ 0.50 m`\n\n"
            "## Structure\n\n"
            "```text\n"
            f"{out_dir.name}/\n"
            "├── README.md\n"
            "├── README.txt\n"
            "├── camera_intrinsics.json\n"
            "├── metadata.csv\n"
            "├── rgb/frame_XXX.png\n"
            "├── depth_png/frame_XXX.png\n"
            "└── depth_meters/frame_XXX.npy\n"
            "```\n\n"
            "## Which Pose To Use\n\n"
            "이미지 정합/stitching/reconstruction에는 `metadata.csv`의 "
            "`T_world_render_camera_00` ~ `T_world_render_camera_33` 값을 우선 사용하세요. "
            "이 값이 실제 MuJoCo renderer가 사용한 `d405_camera`의 world pose입니다.\n\n"
            "## Metadata Columns\n\n"
            "- `capture_index`: 이 export 안에서의 0-based 순서\n"
            "- `frame`: 파일 번호\n"
            "- `is_separator`: 검은 구분 이미지이면 1, 실제 capture이면 0\n"
            "- `capture_valid`: stitching/reconstruction에 사용할 수 있는 촬영이면 1\n"
            "- `invalid_reason`: 검은 invalid capture frame의 제외 사유\n"
            "- `capture_look_error_deg`, `capture_position_error_mm`: look-at gate 검증 오차\n"
            "- `ring_id`, `ring_name`: multi-ring capture pass 정보\n"
            "- `seam_target_x/y/z`: 이 frame에서 카메라가 바라본 접합부 seam point\n"
            "- `q1` ~ `q6`: robot joint angle [rad]\n"
            "- `T_world_render_camera_*`: 실제 렌더 카메라 pose\n"
            "- `T_world_tcp_*`: TCP pose\n"
            "- `T_world_optical_site_*`: camera optical center site pose\n\n"
            "## Python Load Example\n\n"
            "```python\n"
            "import csv\n"
            "import numpy as np\n"
            "from pathlib import Path\n\n"
            f"dataset = Path('{out_dir.name}')\n"
            "with open(dataset / 'metadata.csv') as f:\n"
            "    row = next(csv.DictReader(f))\n\n"
            "depth = np.load(dataset / row['depth_npy_path'])\n"
            "T_world_camera = np.array([\n"
            "    [float(row[f'T_world_render_camera_{r}{c}']) for c in range(4)]\n"
            "    for r in range(4)\n"
            "])\n"
            f"q = np.array([float(row[f'q{{i}}']) for i in range(1, {NDOF + 1})])\n"
            "```\n"
        )

    print(f"[DATASET] Done: {out_dir}")
    print(f"[DATASET] Metadata: {metadata_path}")
    print(f"[DATASET] Intrinsics: {out_dir / 'camera_intrinsics.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE_XML, help="MuJoCo scene XML path.")
    parser.add_argument("--out", default="inspection_dataset_20", help="Output folder.")
    parser.add_argument("--frames", type=int, default=20, help="Number of synchronized capture sets.")
    parser.add_argument("--start-index", type=int, default=0, help="First frame number used in filenames.")
    parser.add_argument("--multi-ring", action="store_true", help="Use near/nominal/far inspection rings.")
    parser.set_defaults(insert_separators=True)
    parser.add_argument("--insert-separators", dest="insert_separators", action="store_true",
                        help="Insert black separator frames at segment/ring boundaries. Enabled by default.")
    parser.add_argument("--no-insert-separators", dest="insert_separators", action="store_false",
                        help="Do not insert black separator frames at segment/ring boundaries.")
    parser.add_argument("--overlap", type=float, default=None,
                        help="Desired image overlap ratio. If set, frame count is computed automatically.")
    parser.add_argument("--min-frames-per-ring", type=int, default=20,
                        help="Minimum frame count per ring when --overlap is used.")
    parser.add_argument("--camera-name", default="d405_camera", help="MuJoCo camera to render.")
    parser.add_argument("--retries", type=int, default=8, help="IK random restarts per frame.")
    parser.add_argument("--rng-seed", type=int, default=7, help="Deterministic IK seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 2:
        raise ValueError("--frames must be at least 2.")
    if args.min_frames_per_ring < 2:
        raise ValueError("--min-frames-per-ring must be at least 2.")
    if not os.path.exists(args.scene):
        raise FileNotFoundError(args.scene)

    export_dataset(
        scene_path=args.scene,
        out_dir=Path(args.out),
        n_frames=args.frames,
        start_index=args.start_index,
        camera_name=args.camera_name,
        retries=args.retries,
        rng_seed=args.rng_seed,
        multi_ring=args.multi_ring,
        overlap=args.overlap,
        min_frames_per_ring=args.min_frames_per_ring,
        insert_separators=args.insert_separators,
    )


if __name__ == "__main__":
    main()
