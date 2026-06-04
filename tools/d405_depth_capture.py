#!/usr/bin/env python3
"""Render and visualize a RealSense D405-style depth image from MuJoCo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.franka_ik_solver import set_arm_qpos, solve_trajectory
from mujoco_viewer import (
    EE_LOOK_AXIS_SIGN,
    FLANGE_CENTER,
    FRANKA_READY,
    SCENE_XML,
    camera_poses_to_tcp_poses,
    generate_segmented_reference,
)


D405_WIDTH = 1280
D405_HEIGHT = 800
D405_MIN_Z = 0.07
D405_MAX_Z = 0.50
D405_VERTICAL_FOV_DEG = 58.0
D405_HORIZONTAL_FOV_DEG = 87.0


def set_d405_clipping(model: mujoco.MjModel) -> None:
    """Set MuJoCo near/far clipping to the D405 ideal range in meters."""
    extent = max(float(model.stat.extent), 1e-12)
    model.vis.map.znear = D405_MIN_Z / extent
    model.vis.map.zfar = D405_MAX_Z / extent


def raw_depth_buffer_to_meters(depth_buffer: np.ndarray, znear: float, zfar: float) -> np.ndarray:
    """
    Convert MuJoCo/OpenGL reverse-Z depth-buffer values to metric depth.

    The official mujoco.Renderer does this internally when
    renderer.enable_depth_rendering() is used. This function is provided for
    explicit documentation or for low-level mjr_readPixels depth buffers.
    """
    z = np.asarray(depth_buffer, dtype=np.float64)
    znear32 = np.float32(znear)
    zfar32 = np.float32(zfar)

    denom_range = zfar32 - znear32
    if abs(float(denom_range)) < 1e-12:
        raise ValueError("znear and zfar must be different.")

    c_coef = -(zfar32 + znear32) / denom_range
    d_coef = -(np.float32(2.0) * zfar32 * znear32) / denom_range

    # MuJoCo 3.x uses reverse-Z for depth rendering.
    c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
    d_coef = np.float32(-0.5) * d_coef

    denom = z + c_coef
    meters = np.full_like(z, np.nan, dtype=np.float64)
    valid = np.isfinite(z) & (np.abs(denom) > 1e-12)
    meters[valid] = d_coef / denom[valid]
    return meters.astype(np.float32)


def mask_d405_range(depth_m: np.ndarray) -> np.ma.MaskedArray:
    """Mask invalid, infinite, and out-of-D405-range depth values."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    invalid = (
        ~np.isfinite(depth_m)
        | (depth_m <= 0.0)
        | (depth_m < D405_MIN_Z)
        | (depth_m > D405_MAX_Z)
    )
    return np.ma.array(depth_m, mask=invalid)


def d405_horizontal_fov_from_vertical(width: int, height: int, fovy_deg: float) -> float:
    aspect = float(width) / float(height)
    fovy = np.deg2rad(fovy_deg)
    fovx = 2.0 * np.arctan(aspect * np.tan(0.5 * fovy))
    return float(np.rad2deg(fovx))


def vertical_fov_from_horizontal(width: int, height: int, fovx_deg: float) -> float:
    aspect = float(width) / float(height)
    fovx = np.deg2rad(fovx_deg)
    fovy = 2.0 * np.arctan(np.tan(0.5 * fovx) / aspect)
    return float(np.rad2deg(fovy))


def set_inspection_waypoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    waypoint_index: int,
    retries: int,
) -> None:
    """Move Franka to one segmented inspection waypoint before rendering."""
    _, _, _, _, camera_poses, look_targets = generate_segmented_reference(return_targets=True)
    waypoint_index = int(np.clip(waypoint_index, 0, len(camera_poses) - 1))
    tcp_poses = camera_poses_to_tcp_poses([camera_poses[waypoint_index]])
    q_hist, flags = solve_trajectory(
        model,
        data,
        mujoco,
        tcp_poses,
        look_target=look_targets[waypoint_index],
        q_start=FRANKA_READY,
        retries=retries,
        verbose=False,
        axis_col=2,
        axis_sign=EE_LOOK_AXIS_SIGN,
    )
    if not flags[0]:
        raise RuntimeError(f"IK failed for inspection waypoint {waypoint_index}.")
    set_arm_qpos(model, data, mujoco, q_hist[0])


def render_depth_meters(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    """Render a metric depth map from the named MuJoCo camera."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found in model.")

    model.cam_fovy[cam_id] = D405_VERTICAL_FOV_DEG
    set_d405_clipping(model)

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)

    # MuJoCo 3.4.0 Renderer returns metric depth after internal linearization.
    depth_m = renderer.render().astype(np.float32)
    renderer.close()
    return depth_m


def visualize_depth(depth_m: np.ndarray, save_path: str | None, show: bool) -> None:
    depth_masked = mask_d405_range(depth_m)
    valid = depth_masked.compressed()

    if valid.size:
        print(
            "[D405 depth] valid pixels: "
            f"{valid.size}/{depth_m.size}, "
            f"min={valid.min():.4f} m, max={valid.max():.4f} m, mean={valid.mean():.4f} m"
        )
    else:
        print("[D405 depth] no valid pixels inside 0.07 m ~ 0.50 m.")

    fig, ax = plt.subplots(figsize=(12.8, 8.0))
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad(color="black")
    im = ax.imshow(
        depth_masked,
        cmap=cmap,
        vmin=D405_MIN_Z,
        vmax=D405_MAX_Z,
        interpolation="nearest",
    )
    ax.set_title("Intel RealSense D405 Depth Map")
    ax.set_xlabel("pixel u")
    ax.set_ylabel("pixel v")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("depth [m]")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160)
        print(f"[D405 depth] saved visualization: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE_XML, help="MuJoCo XML path.")
    parser.add_argument("--camera-name", default="d405_camera")
    parser.add_argument("--width", type=int, default=D405_WIDTH)
    parser.add_argument("--height", type=int, default=D405_HEIGHT)
    parser.add_argument("--waypoint-index", type=int, default=0, help="Inspection waypoint to render.")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--no-ik", action="store_true", help="Render current XML/default pose instead of IK waypoint.")
    parser.add_argument("--save", default="outputs/depth/d405_depth_map.png", help="Path for matplotlib visualization.")
    parser.add_argument("--save-npy", default="outputs/depth/d405_depth_meters.npy", help="Path for raw metric depth .npy.")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.scene):
        raise FileNotFoundError(args.scene)

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not args.no_ik:
        set_inspection_waypoint_pose(model, data, args.waypoint_index, args.retries)

    depth_m = render_depth_meters(
        model,
        data,
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
    )

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.save_npy, depth_m)
    print(f"[D405 depth] saved metric depth array: {args.save_npy}")
    print(
        "[D405 depth] renderer: "
        f"{args.width}x{args.height}, znear={D405_MIN_Z:.2f} m, zfar={D405_MAX_Z:.2f} m, "
        f"vertical_fov={D405_VERTICAL_FOV_DEG:.1f} deg"
    )
    print(
        "[D405 depth] pinhole horizontal FOV implied by 1280x800 and fovy=58: "
        f"{d405_horizontal_fov_from_vertical(args.width, args.height, D405_VERTICAL_FOV_DEG):.2f} deg"
    )
    print(
        "[D405 depth] vertical FOV required to force 87 deg horizontal at 1280x800: "
        f"{vertical_fov_from_horizontal(args.width, args.height, D405_HORIZONTAL_FOV_DEG):.2f} deg"
    )
    visualize_depth(depth_m, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
