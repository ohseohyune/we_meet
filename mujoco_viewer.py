"""
mujoco_viewer.py — Visualize robot, pipe-flange assembly, and inspection trajectory.

Usage
-----
  python mujoco_viewer.py                  # interactive viewer, no IK
  python mujoco_viewer.py --ik             # compute IK and animate trajectory
  python mujoco_viewer.py --camera         # render from d405_camera viewpoint
  python mujoco_viewer.py --save-video     # save trajectory video (requires ffmpeg)
  python mujoco_viewer.py --waypoints      # print waypoint table and exit

Requirements
------------c
  pip install mujoco numpy

MuJoCo version assumed: ≥ 3.0 (uses mujoco.viewer passive_viewer API).
If using MuJoCo 2.x, replace the viewer section with mujoco.viewer.launch().

Assumptions
-----------
- scene.xml is in the same directory as this script.
- IK solutions are seeded sequentially; first seed is q=[0,0,0,0,0,0,0].
- Trajectory animation speed controlled by ANIMATION_SPEED (waypoints/sec).
- Camera renders at 640×480 by default.
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# ── MuJoCo import ────────────────────────────────────────────────────────────
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("[WARNING] mujoco not installed. Install with: pip install mujoco")
    print("          Trajectory math and IK will still work without it.\n")

from trajectory.generator import (
    FLANGE_CENTER, SEAM_RADIUS, STANDOFF,
    PIPE_OFFSET_X, PIPE_LENGTH, PIPE_HEIGHT,
    print_waypoints, rot_to_quat, look_at_rotation, make_pose,
)
from trajectory.circle import (
    DEFAULT_MULTI_RING_SPECS,
    SEGMENTS,
    estimate_multi_ring_frames,
    multi_ring_segmented_trajectory,
    segmented_circle_trajectory,
)
from control.franka_ik_solver import joint_limits, solve_trajectory, set_arm_qpos, site_pose


# ── Configuration ─────────────────────────────────────────────────────────────
SCENE_XML        = os.path.join(os.path.dirname(__file__), "scene.xml")
ANIMATION_SPEED  = 1.0    # trajectory time multiplier
D405_DEPTH_WIDTH  = 1280
D405_DEPTH_HEIGHT = 800
D405_MIN_Z = 0.07
D405_MAX_Z = 0.50
D405_VERTICAL_FOV_DEG = 58.0
CAMERA_WIDTH     = D405_DEPTH_WIDTH
CAMERA_HEIGHT    = D405_DEPTH_HEIGHT
IK_SEED          = np.array([0.5, -0.8, 0.0, 1.2, 0.0, 0.8, 0.0])  # rough inspection start
CAMERA_OFFSET_TCP_Z = 0.10  # camera optical center is 100 mm along TCP -z
NDOF = 7
FRANKA_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0])
TRAJECTORY_CENTER = np.array([0.37, 0.0, 0.57])
TRAJECTORY_RADIUS = 0.1652173913
SEAM_TARGET_RADIUS = SEAM_RADIUS
MULTI_RING_SPECS = DEFAULT_MULTI_RING_SPECS
SEGMENT_DURATION = 9.0
TRAJECTORY_DT = 0.02
IK_WAYPOINTS = 240
APPROACH_DURATION = 9.0
EE_LOOK_AXIS_SIGN = 1.0  # +z axis points toward the circle inside.


# ── Trajectory markers ────────────────────────────────────────────────────────

def update_trajectory_markers(model, data, positions):
    """Set trajectory site positions in MuJoCo data, sampled over the full path."""
    positions = np.asarray(positions, dtype=float)
    marker_ids = []
    for i in range(100):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"wp_{i:02d}")
        if site_id < 0:
            break
        marker_ids.append(site_id)

    if not marker_ids or len(positions) == 0:
        return

    sample_idx = np.linspace(0, len(positions) - 1, len(marker_ids)).round().astype(int)
    for site_id, pos_idx in zip(marker_ids, sample_idx):
        model.site_pos[site_id] = positions[pos_idx]


def _sample_trajectory(traj, max_waypoints):
    if max_waypoints is None or len(traj["positions"]) <= max_waypoints:
        return traj
    sample_idx = np.linspace(0, len(traj["positions"]) - 1, max_waypoints).round().astype(int)
    sampled = {}
    for key, value in traj.items():
        if isinstance(value, np.ndarray) and len(value) == len(traj["positions"]):
            sampled[key] = value[sample_idx]
        else:
            sampled[key] = value
    return sampled


def generate_segmented_reference(
    max_waypoints=IK_WAYPOINTS,
    return_targets=False,
    multi_ring=False,
    return_info=False,
):
    """Generate the requested top-bottom-top-bottom-top segmented seam trajectory."""
    if multi_ring:
        traj = multi_ring_segmented_trajectory(
            seam_center=FLANGE_CENTER,
            ring_specs=MULTI_RING_SPECS,
            segment_duration=SEGMENT_DURATION,
            dt=TRAJECTORY_DT,
            target_radius=SEAM_TARGET_RADIUS,
        )
    else:
        traj = segmented_circle_trajectory(
            center=TRAJECTORY_CENTER,
            radius=TRAJECTORY_RADIUS,
            segment_duration=SEGMENT_DURATION,
            dt=TRAJECTORY_DT,
            orientation_target=FLANGE_CENTER,
            target_radius=SEAM_TARGET_RADIUS,
        )
    traj = _sample_trajectory(traj, max_waypoints)
    time_values = traj["time"]
    angles = traj["angles"]
    positions = traj["positions"]
    targets = traj["targets"]

    rotations = []
    poses = []
    for pos, target in zip(positions, targets):
        R = look_at_rotation(pos, target)
        rotations.append(R)
        poses.append(make_pose(pos, R))

    output = [time_values, angles, positions, rotations, poses]
    if return_targets:
        output.append(targets)
    if return_info:
        output.append(traj)
    return tuple(output)


# ── Draw coordinate frame ─────────────────────────────────────────────────────

def draw_frame_axes(viewer, pos, R, size=0.08, alpha=0.9):
    """Draw X (red), Y (green), Z (blue) axes at pos with orientation R using viewer geoms."""
    colors = [(1,0,0,alpha), (0,1,0,alpha), (0,0,1,alpha)]
    for i in range(3):
        axis_end = pos + size * R[:, i]
        viewer.add_marker(
            pos=pos, size=[0.004, 0.004, size],
            mat=R, rgba=colors[i], type=mujoco.mjtGeom.mjGEOM_ARROW,
            label=""
        )


# ── IK and trajectory computation ────────────────────────────────────────────

def camera_poses_to_tcp_poses(camera_poses):
    """
    Convert desired camera optical-center poses to TCP poses.

    The Franka XML mounts the camera 100 mm along TCP -z:
      p_camera = p_tcp - 0.10 * z_tcp.
    For inside-facing EE motion, TCP +z points toward the current seam target, so:
      p_tcp = p_camera + 0.10 * z_inside.
    """
    tcp_poses = []
    for T_cam in camera_poses:
        T_tcp = T_cam.copy()
        T_tcp[:3, 3] = T_cam[:3, 3] + EE_LOOK_AXIS_SIGN * CAMERA_OFFSET_TCP_Z * T_cam[:3, 2]
        T_tcp[:3, 0] = T_cam[:3, 0]
        T_tcp[:3, 1] = T_cam[:3, 1]
        T_tcp[:3, 2] = EE_LOOK_AXIS_SIGN * T_cam[:3, 2]
        tcp_poses.append(T_tcp)
    return tcp_poses


def compute_ik_trajectory(model, data, verbose=True, retries=16, rng_seed=7):
    """Compute joint trajectory for the segmented flange inspection path."""
    print("\n[IK] Generating inspection trajectory...")
    (
        time_values,
        angles,
        camera_positions,
        camera_rotations,
        camera_poses,
        look_targets,
        traj_info,
    ) = generate_segmented_reference(
        return_targets=True,
        return_info=True,
    )
    tcp_poses_world = camera_poses_to_tcp_poses(camera_poses)

    if verbose:
        print_waypoints(angles, camera_positions, camera_rotations)
        print(f"[IK] Camera offset from TCP: {CAMERA_OFFSET_TCP_Z*1000:.0f} mm along TCP -z")

    print(f"[IK] Solving Franka MuJoCo look-at IK for {len(tcp_poses_world)} waypoints...")
    Q, flags = solve_trajectory(
        model,
        data,
        mujoco,
        tcp_poses_world,
        look_target=look_targets,
        q_start=FRANKA_READY.copy(),
        retries=retries,
        rng_seed=rng_seed,
        verbose=verbose,
        axis_col=2,
        axis_sign=EE_LOOK_AXIS_SIGN,
    )

    n_ok   = sum(flags)
    n_fail = len(flags) - n_ok
    print(f"\n[IK] Results: {n_ok}/{len(flags)} converged, {n_fail} failed")

    if n_fail > 0:
        failed = [i+1 for i, ok in enumerate(flags) if not ok]
        print(f"     Failed waypoints: {failed}")
        print("     Tip: increase --retries or adjust IK_SEED/STANDOFF.")

    return Q, flags, time_values, angles, camera_positions, camera_rotations, camera_poses, tcp_poses_world, traj_info


# ── FK verification ──────────────────────────────────────────────────────────

def verify_ik(Q, tcp_poses, camera_poses=None):
    """Print TCP and camera position error for each waypoint after IK."""
    print("\n[FK] IK verification:")
    max_tcp_err = 0.0
    max_cam_err = 0.0
    for i, (q, T_des) in enumerate(zip(Q, tcp_poses)):
        set_arm_qpos(verify_ik.model, verify_ik.data, mujoco, q)
        T_act = site_pose(verify_ik.model, verify_ik.data, mujoco, "tcp")
        err = np.linalg.norm(T_des[:3,3] - T_act[:3,3])
        max_tcp_err = max(max_tcp_err, err)
        if err > 1e-3:
            print(f"  [WARN] WP {i+1:2d}: tcp_pos_err = {err*1000:.2f} mm")
        if camera_poses is not None:
            T_cam_act = site_pose(verify_ik.model, verify_ik.data, mujoco, "camera_optical_center")
            cam_err = np.linalg.norm(camera_poses[i][:3, 3] - T_cam_act[:3, 3])
            max_cam_err = max(max_cam_err, cam_err)
    print(f"  Max TCP position error   : {max_tcp_err*1000:.3f} mm")
    if camera_poses is not None:
        print(f"  Max camera position error: {max_cam_err*1000:.3f} mm")


def save_trajectory_csv(path, Q, flags, angles, camera_poses, tcp_poses):
    """Save desired EE poses and IK joint trajectory in one CSV file."""
    fieldnames = [
        "index", "angle_deg", "success",
        "camera_x", "camera_y", "camera_z", "camera_qw", "camera_qx", "camera_qy", "camera_qz",
        "tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz",
        "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (q, ok, phi, T_cam, T_tcp) in enumerate(zip(Q, flags, angles, camera_poses, tcp_poses)):
            q_cam = rot_to_quat(T_cam[:3, :3])
            q_tcp = rot_to_quat(T_tcp[:3, :3])
            row = {
                "index": i,
                "angle_deg": np.rad2deg(phi),
                "success": int(ok),
                "camera_x": T_cam[0, 3],
                "camera_y": T_cam[1, 3],
                "camera_z": T_cam[2, 3],
                "camera_qw": q_cam[0],
                "camera_qx": q_cam[1],
                "camera_qy": q_cam[2],
                "camera_qz": q_cam[3],
                "tcp_x": T_tcp[0, 3],
                "tcp_y": T_tcp[1, 3],
                "tcp_z": T_tcp[2, 3],
                "tcp_qw": q_tcp[0],
                "tcp_qx": q_tcp[1],
                "tcp_qy": q_tcp[2],
                "tcp_qz": q_tcp[3],
            }
            row.update({f"q{j+1}": q[j] for j in range(NDOF)})
            writer.writerow(row)
    print(f"[CSV] Wrote trajectory: {path}")


# ── Camera render ─────────────────────────────────────────────────────────────

def mask_d405_depth(depth_m):
    depth_m = np.asarray(depth_m, dtype=np.float32)
    invalid = (
        ~np.isfinite(depth_m)
        | (depth_m <= 0.0)
        | (depth_m < D405_MIN_Z)
        | (depth_m > D405_MAX_Z)
    )
    return np.ma.array(depth_m, mask=invalid)


def save_depth_png(depth_m, path):
    depth_masked = mask_d405_depth(depth_m)
    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad(color="black")
    plt.imsave(
        path,
        depth_masked,
        cmap=cmap,
        vmin=D405_MIN_Z,
        vmax=D405_MAX_Z,
    )


def save_black_depth_png(path):
    plt.imsave(
        path,
        np.zeros((D405_DEPTH_HEIGHT, D405_DEPTH_WIDTH), dtype=np.uint8),
        cmap="gray",
        vmin=0,
        vmax=255,
    )


def set_d405_depth_rendering(model, cam_id):
    extent = max(float(model.stat.extent), 1e-12)
    model.vis.map.znear = D405_MIN_Z / extent
    model.vis.map.zfar = D405_MAX_Z / extent
    model.cam_fovy[cam_id] = D405_VERTICAL_FOV_DEG


def render_inspection_cameras(model, data, Q, traj_info=None, out_dir="inspection_frames", camera_name="d405_camera"):
    """Render D405-style metric depth maps from the selected fixed camera."""
    if not MUJOCO_AVAILABLE:
        return
    os.makedirs(out_dir, exist_ok=True)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        print(f"[WARN] camera '{camera_name}' not found in model.")
        return

    set_d405_depth_rendering(model, cam_id)
    renderer = mujoco.Renderer(model, height=D405_DEPTH_HEIGHT, width=D405_DEPTH_WIDTH)
    renderer.enable_depth_rendering()

    depth_dir = os.path.join(out_dir, "depth_meters")
    png_dir = os.path.join(out_dir, "depth_png")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    segment_ids = None
    separator_count = 0
    if traj_info is not None:
        segment_ids = traj_info.get("segment_id")
        if segment_ids is not None:
            segment_ids = np.asarray(segment_ids)
            separator_count = int(np.count_nonzero(segment_ids[1:] != segment_ids[:-1]))

    total_frames = len(Q) + separator_count
    print(
        f"\n[CAM] Rendering {len(Q)} D405 depth frames from '{camera_name}' "
        f"to '{out_dir}/'..."
    )
    if separator_count:
        print(f"[CAM] Inserting {separator_count} black separator frames at segment boundaries.")

    valid_counts = []
    output_i = 0
    for i, q in enumerate(Q):
        if segment_ids is not None and i > 0 and int(segment_ids[i]) != int(segment_ids[i - 1]):
            depth_m = np.full((D405_DEPTH_HEIGHT, D405_DEPTH_WIDTH), np.nan, dtype=np.float32)
            np.save(os.path.join(depth_dir, f"frame_{output_i:03d}.npy"), depth_m)
            save_black_depth_png(os.path.join(png_dir, f"frame_{output_i:03d}.png"))
            output_i += 1

        set_arm_qpos(model, data, mujoco, q)
        renderer.update_scene(data, camera=cam_id)
        depth_m = renderer.render().astype(np.float32)
        depth_masked = mask_d405_depth(depth_m)
        valid_counts.append(depth_masked.count())

        np.save(os.path.join(depth_dir, f"frame_{output_i:03d}.npy"), depth_m)
        save_depth_png(depth_m, os.path.join(png_dir, f"frame_{output_i:03d}.png"))
        output_i += 1

    renderer.close()
    print(f"[CAM] Saved metric depth arrays: {depth_dir}/frame_*.npy")
    print(f"[CAM] Saved depth visualizations: {png_dir}/frame_*.png")
    if separator_count:
        print(f"[CAM] Frame count: {len(Q)} captures + {separator_count} separators = {total_frames} files")
    print(
        f"[CAM] D405 depth config: {D405_DEPTH_WIDTH}x{D405_DEPTH_HEIGHT}, "
        f"range={D405_MIN_Z:.2f}m~{D405_MAX_Z:.2f}m, fovy={D405_VERTICAL_FOV_DEG:.1f}deg"
    )
    if valid_counts:
        print(
            f"[CAM] Valid depth pixels per frame: "
            f"min={min(valid_counts)}, max={max(valid_counts)}, "
            f"mean={np.mean(valid_counts):.1f}/{D405_DEPTH_WIDTH * D405_DEPTH_HEIGHT}"
        )


# ── Interactive viewer ───────────────────────────────────────────────────────

def interpolate_q(time_values, Q, t):
    if t <= time_values[0]:
        return Q[0]
    if t >= time_values[-1]:
        return Q[-1]
    hi = int(np.searchsorted(time_values, t, side="right"))
    lo = hi - 1
    span = max(time_values[hi] - time_values[lo], 1e-12)
    alpha = (t - time_values[lo]) / span
    return (1.0 - alpha) * Q[lo] + alpha * Q[hi]


def prepend_zero_to_start(model, time_values, Q, approach_duration=APPROACH_DURATION):
    limits = joint_limits(model, mujoco)
    q_zero = np.clip(np.zeros(7), limits[:, 0], limits[:, 1])
    q_start = Q[0]

    if approach_duration <= 0.0:
        return time_values, Q

    nominal_dt = float(np.median(np.diff(time_values))) if len(time_values) > 1 else 0.02
    nominal_dt = max(nominal_dt, 0.01)
    n_steps = max(2, int(np.ceil(approach_duration / nominal_dt)))
    approach_time = np.linspace(0.0, approach_duration, n_steps + 1)
    u = approach_time / approach_duration
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    approach_q = q_zero[None, :] + s[:, None] * (q_start - q_zero)[None, :]

    shifted_time = approach_duration + (time_values - time_values[0])
    return (
        np.concatenate([approach_time, shifted_time[1:]]),
        np.vstack([approach_q, Q[1:]]),
    )


def run_interactive(
    model,
    data,
    Q=None,
    positions=None,
    time_values=None,
    approach_duration=APPROACH_DURATION,
    fixed_camera_name=None,
):
    """Launch interactive MuJoCo viewer, optionally animating trajectory."""
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available; cannot launch viewer.")
        return

    # Set initial pose
    if Q is not None and len(Q) > 0:
        set_arm_qpos(model, data, mujoco, Q[0])

    print("\n[VIEWER] Launching interactive window...")
    print("  Press [Space] to pause/resume animation.")
    print("  Press [Esc]   to exit.\n")

    paused        = [False]
    if time_values is None and Q is not None:
        time_values = np.linspace(0.0, SEGMENT_DURATION * len(SEGMENTS), len(Q))
    if Q is not None:
        time_values, Q = prepend_zero_to_start(
            model,
            time_values,
            Q,
            approach_duration=approach_duration,
        )
        set_arm_qpos(model, data, mujoco, Q[0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if fixed_camera_name:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, fixed_camera_name)
            if cam_id < 0:
                print(f"[WARN] camera '{fixed_camera_name}' not found; using free viewer camera.")
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = cam_id

        if not fixed_camera_name:
            viewer.cam.lookat[:] = FLANGE_CENTER
            viewer.cam.distance   = 1.8
            viewer.cam.elevation  = -25
            viewer.cam.azimuth    = 45

        start_time = time.time()
        while viewer.is_running():
            now = time.time()

            if Q is not None and not paused[0]:
                traj_t = min((now - start_time) * ANIMATION_SPEED, time_values[-1])
                q = interpolate_q(time_values, Q, traj_t)
                set_arm_qpos(model, data, mujoco, q)

            viewer.sync()
            time.sleep(0.002)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Flange Inspection Viewer")
    parser.add_argument("--ik",         action="store_true", help="Compute IK and animate")
    parser.add_argument("--camera",     action="store_true", help="Render frames from the selected fixed camera")
    parser.add_argument("--camera-viewer", action="store_true", help="Open the interactive viewer through the selected fixed camera")
    parser.add_argument("--camera-name", default="d405_camera", help="MuJoCo fixed camera name to render.")
    parser.add_argument("--save-video", action="store_true", help="Save animation as video")
    parser.add_argument("--waypoints",  action="store_true", help="Print waypoints and exit")
    parser.add_argument("--verify",     action="store_true", help="Print IK FK verification")
    parser.add_argument("--no-viewer",  action="store_true", help="Do not open the interactive viewer")
    parser.add_argument("--retries",     type=int, default=16, help="Random restarts per waypoint for IK fallback")
    parser.add_argument("--rng-seed",    type=int, default=7, help="Random seed for deterministic IK retry fallback")
    parser.add_argument("--approach-duration", type=float, default=APPROACH_DURATION,
                        help="Seconds to move from zero pose to the 12 o'clock start")
    parser.add_argument("--export-csv", nargs="?", const="outputs/trajectories/inspection_trajectory.csv",
                        help="Write desired poses and joint trajectory to CSV")
    args = parser.parse_args()

    # ── Waypoints only ────────────────────────────────────────────────────
    if args.waypoints:
        _, angles, positions, rotations, poses = generate_segmented_reference(max_waypoints=36)
        print_waypoints(angles, positions, rotations)
        return

    # ── Load model ───────────────────────────────────────────────────────
    if not os.path.exists(SCENE_XML):
        print(f"[ERROR] scene.xml not found at: {SCENE_XML}")
        sys.exit(1)

    if not MUJOCO_AVAILABLE:
        print("MuJoCo is required for the Franka model viewer and IK.")
        return

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)

    Q = None
    flags = None
    time_values = None
    angles = None
    camera_poses = None
    tcp_poses = None
    positions = None
    traj_info = None

    # ── IK computation ───────────────────────────────────────────────────
    if not (args.ik or args.camera or args.camera_viewer or args.verify or args.export_csv or args.no_viewer):
        args.ik = True

    if args.ik or args.camera or args.camera_viewer or args.verify or args.export_csv:
        Q, flags, time_values, angles, positions, rotations, camera_poses, tcp_poses, traj_info = compute_ik_trajectory(
            model,
            data,
            retries=args.retries,
            rng_seed=args.rng_seed,
        )
        if args.verify:
            verify_ik.model = model
            verify_ik.data = data
            verify_ik(Q, tcp_poses, camera_poses)
        if args.export_csv:
            save_trajectory_csv(args.export_csv, Q, flags, angles, camera_poses, tcp_poses)

    if args.no_viewer and not args.camera and not args.save_video:
        return

    # Place trajectory markers
    if positions is not None:
        update_trajectory_markers(model, data, positions)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ── Camera render ─────────────────────────────────────────────────────
    if args.camera and Q is not None:
        render_inspection_cameras(model, data, Q, traj_info=traj_info, camera_name=args.camera_name)
        if not args.camera_viewer:
            print("[CAM] Camera frames were written to inspection_frames/.")
            print("[CAM] Use --camera-viewer if you want an interactive fixed-camera window.")
            return

    if args.save_video:
        print("[WARN] --save-video is reserved for future video export; use --camera for frame export.")

    if args.no_viewer:
        return

    # ── Interactive viewer ────────────────────────────────────────────────
    run_interactive(
        model,
        data,
        Q=Q,
        positions=positions,
        time_values=time_values,
        approach_duration=args.approach_duration,
        fixed_camera_name=args.camera_name if args.camera_viewer else None,
    )


if __name__ == "__main__":
    main()
