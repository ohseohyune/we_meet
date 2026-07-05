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
- IK solutions are seeded sequentially; first seed is q=[0,0,0,0,0,0].
- Trajectory animation speed controlled by ANIMATION_SPEED (waypoints/sec).
- Camera renders at 640×480 by default.
"""

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

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
    generate_waypoints, seam_target_position,
)
from trajectory.circle import (
    DEFAULT_MULTI_RING_SPECS,
    SEGMENTS,
    estimate_multi_ring_frames,
    multi_ring_segmented_trajectory,
    segmented_circle_trajectory,
)
from control.franka_ik_solver import (
    evaluate_look_at_trajectory,
    evaluate_collision_trajectory,
    joint_motion_metrics,
    joint_limits,
    retime_joint_trajectory,
    solve_trajectory,
    set_arm_qpos,
    site_pose,
)


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
IK_SEED          = np.array([0.5, -0.8, 0.0, 1.2, 0.0, 0.8])  # rough inspection start
NDOF = 6
ROBOT_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571])
RIGHT_BIASED_READY = np.array([1.50, -0.785, 0.40, -2.356, 0.0, 1.571])
# Weld-specific seed: wrist branch with ample j4 margin (old seed had q4=-3.142, at the
# -pi limit, forcing a 3.6 rad branch flip near phi 64°).
# Derived from IK solution at phi=63.0° with pipe at x=0.8475 (1.5x distance).
WELD_BIASED_READY = np.array([0.295, -2.541, 2.804, 0.558, 1.321, -1.237])
# Left-half (via 9 o'clock) seed: the right-half branch dies just past 12 o'clock
# (j4 limit), so the left half needs its own wrist branch. Seeding seg3 with it pins
# the single unavoidable branch change at the 12 o'clock pass boundary instead of
# mid-seam. Derived from IK solution at phi=189° with pipe at x=0.6500.
WELD_LEFT_HALF_SEED = np.array([-0.275, -2.039, 1.972, -2.652, 2.228, -1.123])
TRAJECTORY_CENTER = FLANGE_CENTER + np.array([-0.3000, 0.0, 0.0])
TRAJECTORY_RADIUS = 0.1200
SEAM_TARGET_RADIUS = SEAM_RADIUS
MULTI_RING_SPECS = DEFAULT_MULTI_RING_SPECS
USE_FEASIBLE_CAPTURE_ARCS = False
PLAYBACK_CAPTURE_ONLY = False
SEGMENT_DURATION = 9.0
TRAJECTORY_DT = 0.02
IK_WAYPOINTS = 216
APPROACH_DURATION = 9.0
MAX_JOINT_SPEED = 0.85
PLAYBACK_MAX_JOINT_ACCEL = 4.00
PLAYBACK_MAX_JOINT_JERK = 60.00
CAPTURE_MAX_LOOK_DEG = 5.0
CAPTURE_MAX_POS_ERR = 0.010
COLLISION_MARGIN = 0.0
COLLISION_AVOIDANCE_PENALTY = 0.0
COLLISION_AVOIDANCE_MAX_JOINT_STEP = 0.75
ENFORCE_CAPTURE_GATE_IN_IK = True
PLAYBACK_NOMINAL_DT = 0.05
PLAYBACK_BRIDGE_STEPS = 80
PLAYBACK_MAX_SAMPLE_STEP = 0.035
PLAYBACK_CARTESIAN_WAYPOINTS = 120
PLAYBACK_CARTESIAN_RETRIES = 1
DEFAULT_IK_RETRIES = 4
COLLISION_AVOIDANCE_DEFAULT_PENALTY = 100.0
SMOOTH_PLAYBACK_INTERPOLATION = False
EE_LOOK_AXIS_COL = 2
EE_LOOK_AXIS_SIGN = -1.0  # MuJoCo camera optical axis is camera-site -z.
TCP_SITE_NAME = "tcp"
CAMERA_SITE_NAME = "camera_optical_center"
CAMERA_OFFSET_FROM_TIP_EE = np.array([-0.00841, -0.03657, 0.08456])
CAMERA_IN_EE_ROT = np.array(
    [
        [0.99463114, 0.01241430, -0.10273648],
        [-0.04061714, 0.95994352, -0.27723400],
        [0.09517956, 0.27991844, 0.95529395],
    ]
)
RIGHT_POSTURE_BIAS = {
    "q_ref": RIGHT_BIASED_READY,
    "body_names": ("link4", "link5", "link6", "ee"),
    "side_axis": 1,
    "side_sign": -1.0,
    "side_margin": 0.10,
    "q_weight": 0.010,
    "body_weight": 0.22,
}
RIGHT_POSTURE_SEGMENT_WEIGHTS = {
    1: 1.5,
    2: 0.45,
    3: 0.25,
    4: 0.20,
}
# ── Welding IK constants ──────────────────────────────────────────────────────
WELD_LOOK_AXIS_COL  = 2      # EE +Z = torch direction
WELD_LOOK_AXIS_SIGN = 1.0    # torch +Z points toward weld (inspection uses -1.0)
WELD_DEFAULT_RETRIES = 16    # tcp site는 camera site보다 workspace 제약이 강해 더 많은 retry 필요
# 위쪽에서 용접: 팔을 +Z(위) 방향으로 편향 (상단 반원 궤적과 함께 사용)
WELD_ABOVE_POSTURE_BIAS = {
    "q_ref": WELD_BIASED_READY,
    "body_names": ("link4", "link5", "link6", "ee"),
    "side_axis": 2,      # Z축 기준
    "side_sign": 1.0,    # +Z 위쪽으로 팔 편향
    "side_margin": 0.10,
    "q_weight": 0.05,
    "body_weight": 0.22,
}
_VIEWER_ROOT = os.path.dirname(os.path.abspath(__file__))
# PLY 캡처 당시의 pipe_flange_assembly 위치 — seam 정합 기준점.
WELD_PLY_CAPTURE_PIPE_POS = np.array([0.5650, 0.0, 0.5700])
WELD_PLY_PATH = os.path.join(_VIEWER_ROOT, "inspection_frames", "output",
                              "reconstructed_pointcloud.ply")

SHOW_TRAJECTORY_MARKERS = False
TRAJECTORY_MARKER_RGBA = np.array([0.0, 0.85, 1.0, 1.0])
TRAJECTORY_MARKER_SIZE = 0.014
FRAME_AXIS_COLORS = (
    np.array([1.0, 0.05, 0.03, 0.92]),
    np.array([0.05, 0.75, 0.08, 0.92]),
    np.array([0.10, 0.25, 1.00, 0.92]),
)
WORLD_FRAME_ORIGIN = np.array([0.0, 0.0, 0.12])
PIPE_FLANGE_FRAME_R = np.column_stack(
    [
        np.array([-1.0, 0.0, 0.0]),  # flange face normal points toward the robot
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
)


# ── Trajectory markers ────────────────────────────────────────────────────────

def _trajectory_marker_ids(model):
    marker_ids = []
    for i in range(100):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"wp_{i:02d}")
        if site_id < 0:
            break
        marker_ids.append(site_id)
    return marker_ids


def hide_trajectory_markers(model):
    """Hide the reference trajectory sites without removing them from scene.xml."""
    for site_id in _trajectory_marker_ids(model):
        model.site_rgba[site_id] = np.array([0.0, 0.0, 0.0, 0.0])
        model.site_size[site_id] = 1.0e-4


def update_trajectory_markers(model, data, positions):
    """Set trajectory site positions in MuJoCo data, sampled over the full path."""
    if not SHOW_TRAJECTORY_MARKERS:
        hide_trajectory_markers(model)
        return

    positions = np.asarray(positions, dtype=float)
    marker_ids = _trajectory_marker_ids(model)

    if not marker_ids or len(positions) == 0:
        return

    sample_idx = np.linspace(0, len(positions) - 1, len(marker_ids)).round().astype(int)
    for site_id, pos_idx in zip(marker_ids, sample_idx):
        model.site_pos[site_id] = positions[pos_idx]
        model.site_rgba[site_id] = TRAJECTORY_MARKER_RGBA
        model.site_size[site_id] = TRAJECTORY_MARKER_SIZE


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


def right_posture_weights(traj_info, n_waypoints):
    """Bias segment 1 strongly to a right-side body posture, then taper."""
    if not traj_info:
        return np.zeros(n_waypoints, dtype=float)
    segment_ids = np.asarray(
        traj_info.get("base_segment_id", traj_info.get("segment_id", [])),
        dtype=int,
    )
    if segment_ids.shape != (n_waypoints,):
        return np.zeros(n_waypoints, dtype=float)
    return np.array(
        [RIGHT_POSTURE_SEGMENT_WEIGHTS.get(int(segment_id), 0.0) for segment_id in segment_ids],
        dtype=float,
    )


def generate_segmented_reference(
    max_waypoints=IK_WAYPOINTS,
    return_targets=False,
    multi_ring=False,
    return_info=False,
):
    """Generate uniform-density circular inspection trajectory.

    circle.py의 quintic-spline segmented 방식은 모든 segment가 phi=90°(12시)에서
    시작·종료하므로 해당 구간에 프레임이 과밀하게 집중되어 reconstruction에서
    뭉개짐이 발생한다. generator.py의 linspace 기반 균일 샘플링으로 교체한다.
    """
    angles, positions, rotations, poses = generate_waypoints(
        n=max_waypoints,
        exclude_bottom=True,
    )
    n = len(angles)
    time_values = np.linspace(0.0, (n - 1) * TRAJECTORY_DT, n)

    output = [time_values, angles, positions, rotations, poses]
    if return_targets:
        targets = [seam_target_position(phi) for phi in angles]
        output.append(targets)
    if return_info:
        # 단일 segment — right_posture_weights가 segment_id=1 가중치(1.5)를 전체에 적용
        output.append({"segment_id": np.ones(n, dtype=int)})
    return tuple(output)


# ── Draw coordinate frame ─────────────────────────────────────────────────────

def _arrow_orientation_from_axis(axis):
    """Build a marker rotation whose local +Z points along `axis`."""
    z = np.asarray(axis, dtype=float).reshape(3)
    z /= np.linalg.norm(z) + 1e-12
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _clear_viewer_overlay(viewer):
    if hasattr(viewer, "user_scn"):
        viewer.user_scn.ngeom = 0


def _add_viewer_marker(viewer, *, pos, size, mat, rgba, type, label=""):
    """Add a viewer marker across MuJoCo viewer API variants."""
    if hasattr(viewer, "add_marker"):
        viewer.add_marker(pos=pos, size=size, mat=mat, rgba=rgba, type=type, label=label)
        return

    if not hasattr(viewer, "user_scn"):
        return
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type,
        np.asarray(size, dtype=float).reshape(3),
        np.asarray(pos, dtype=float).reshape(3),
        np.asarray(mat, dtype=float).reshape(9),
        np.asarray(rgba, dtype=float).reshape(4),
    )
    if label:
        geom.label = str(label)
    scene.ngeom += 1


def draw_frame_axes(viewer, pos, R, size=0.08, alpha=0.9, label=""):
    """Draw X (red), Y (green), Z (blue) axes at pos with orientation R."""
    pos = np.asarray(pos, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3, 3)
    colors = [c.copy() for c in FRAME_AXIS_COLORS]
    for color in colors:
        color[3] = alpha
    for i in range(3):
        _add_viewer_marker(
            viewer,
            pos=pos + 0.5 * size * R[:, i],
            size=[0.006, 0.006, size],
            mat=_arrow_orientation_from_axis(R[:, i]),
            rgba=colors[i],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            label=""
        )
    if label:
        _add_viewer_marker(
            viewer,
            pos=pos,
            size=[0.018, 0.018, 0.018],
            rgba=[1.0, 1.0, 1.0, 0.45],
            mat=np.eye(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            label=label,
        )


def draw_inspection_frames(viewer, model, data, show_tcp=True):
    """Draw presentation frames inside the interactive MuJoCo viewer."""
    _clear_viewer_overlay(viewer)
    draw_frame_axes(viewer, WORLD_FRAME_ORIGIN, np.eye(3), size=0.12, alpha=0.95, label="World/Base")


def draw_weld_trajectory_overlay(viewer, weld_poses, every=3):
    _clear_viewer_overlay(viewer)
    draw_frame_axes(viewer, WORLD_FRAME_ORIGIN, np.eye(3), size=0.12, alpha=0.95, label="World/Base")


# ── IK and trajectory computation ────────────────────────────────────────────

def camera_poses_to_tcp_poses(camera_poses):
    """
    Convert desired camera optical-center poses to MuJoCo site target poses.

    `camera_optical_center` is offset from the EE/tip origin in the EE frame.
    Its site position is already the desired camera center; its site orientation
    follows EE orientation, while the fixed MuJoCo camera optical axis follows EE -x.
    """
    tcp_poses = []
    for T_cam in camera_poses:
        T_tcp = T_cam.copy()
        T_tcp[:3, 3] = T_cam[:3, 3]
        T_tcp[:3, :3] = T_cam[:3, :3]
        tcp_poses.append(T_tcp)
    return tcp_poses


def target_rotation_error_degrees(model, data, Q, target_poses, site_name=CAMERA_SITE_NAME):
    """Return site-vs-target orientation error angle in degrees for each sample."""
    errors = []
    for q, T_target in zip(Q, target_poses):
        set_arm_qpos(model, data, mujoco, q)
        T_actual = site_pose(model, data, mujoco, site_name)
        R_err = T_target[:3, :3] @ T_actual[:3, :3].T
        cos_angle = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
        errors.append(np.rad2deg(np.arccos(cos_angle)))
    return np.asarray(errors, dtype=float)


def max_adjacent_joint_step(Q):
    Q = np.asarray(Q, dtype=float)
    if len(Q) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(Q, axis=0))))


def compute_ik_trajectory(model, data, verbose=True, retries=16, rng_seed=7, max_waypoints=IK_WAYPOINTS,
                          origin_shift=None, ik_progress_cb=None):
    """Compute joint trajectory for the segmented flange inspection path.

    origin_shift: optional (3,) translation applied to the whole reference
    trajectory (camera positions/poses and look targets). Used when the pipe
    is moved away from the position the trajectory constants assume.
    """
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
        max_waypoints=max_waypoints,
        return_targets=True,
        return_info=True,
    )

    if origin_shift is not None:
        shift = np.asarray(origin_shift, dtype=float).reshape(3)
        if np.linalg.norm(shift) > 1e-12:
            camera_positions = np.asarray(camera_positions, dtype=float) + shift
            look_targets = np.asarray(look_targets, dtype=float) + shift
            shifted_poses = []
            for T in camera_poses:
                T2 = np.array(T, dtype=float, copy=True)
                T2[:3, 3] += shift
                shifted_poses.append(T2)
            camera_poses = shifted_poses
            print(f"[IK] Reference trajectory shifted by {np.round(shift, 4)} m")

    tcp_poses_world = camera_poses_to_tcp_poses(camera_poses)

    if verbose:
        print_waypoints(angles, camera_positions, camera_rotations)
        print(
            "[IK] Camera center offset from EE tip: "
            f"{CAMERA_OFFSET_FROM_TIP_EE * 1000.0} mm"
        )

    posture_weights = right_posture_weights(traj_info, len(tcp_poses_world))

    def solve_candidate(label, collision_penalty):
        print(
            f"[IK] Solving 6-DOF MuJoCo look-at IK for {len(tcp_poses_world)} "
            f"waypoints ({label})..."
        )
        return solve_trajectory(
            model,
            data,
            mujoco,
            tcp_poses_world,
            look_target=look_targets,
            q_start=RIGHT_BIASED_READY.copy(),
            retries=retries,
            rng_seed=rng_seed,
            verbose=verbose,
            axis_col=EE_LOOK_AXIS_COL,
            axis_sign=EE_LOOK_AXIS_SIGN,
            posture_bias=RIGHT_POSTURE_BIAS,
            posture_weights=posture_weights,
            max_capture_look_deg=CAPTURE_MAX_LOOK_DEG,
            max_capture_pos_err=CAPTURE_MAX_POS_ERR,
            invalid_candidate_penalty=1.0e6 if ENFORCE_CAPTURE_GATE_IN_IK else 0.0,
            collision_penalty=collision_penalty,
            collision_margin=COLLISION_MARGIN,
            site_name=CAMERA_SITE_NAME,
            progress_cb=ik_progress_cb,
        )

    def evaluate_candidate(Q_candidate):
        capture = evaluate_look_at_trajectory(
            model,
            data,
            mujoco,
            Q_candidate,
            tcp_poses_world,
            look_targets,
            axis_col=EE_LOOK_AXIS_COL,
            axis_sign=EE_LOOK_AXIS_SIGN,
            site_name=CAMERA_SITE_NAME,
            max_pos_err=CAPTURE_MAX_POS_ERR,
            max_look_deg=CAPTURE_MAX_LOOK_DEG,
        )
        collision = evaluate_collision_trajectory(
            model,
            data,
            mujoco,
            Q_candidate,
            collision_margin=COLLISION_MARGIN,
        )
        rotation_deg = target_rotation_error_degrees(
            model,
            data,
            Q_candidate,
            tcp_poses_world,
            site_name=CAMERA_SITE_NAME,
        )
        return capture, collision, rotation_deg

    Q, flags = solve_candidate("smooth tracking", collision_penalty=0.0)
    capture_metrics, collision_metrics, target_rot_deg = evaluate_candidate(Q)
    avoidance_status = "not_needed"
    smooth_collision_frames = int(np.count_nonzero(~collision_metrics["collision_free"]))
    if smooth_collision_frames and COLLISION_AVOIDANCE_PENALTY > 0.0:
        Q_avoid, flags_avoid = solve_candidate(
            "collision-aware candidate",
            collision_penalty=COLLISION_AVOIDANCE_PENALTY,
        )
        avoid_capture_metrics, avoid_collision_metrics, avoid_target_rot_deg = evaluate_candidate(Q_avoid)
        avoid_collision_frames = int(np.count_nonzero(~avoid_collision_metrics["collision_free"]))
        avoid_step = max_adjacent_joint_step(Q_avoid)
        avoid_capture_ok = bool(np.all(avoid_capture_metrics["capture_valid"]))
        improves_collision = avoid_collision_frames < smooth_collision_frames
        continuous_enough = avoid_step <= COLLISION_AVOIDANCE_MAX_JOINT_STEP
        if avoid_capture_ok and improves_collision and continuous_enough:
            Q = Q_avoid
            flags = flags_avoid
            capture_metrics = avoid_capture_metrics
            collision_metrics = avoid_collision_metrics
            target_rot_deg = avoid_target_rot_deg
            avoidance_status = "accepted"
            print(
                f"[IK] Using collision-aware candidate "
                f"(collisions {smooth_collision_frames} -> {avoid_collision_frames}, "
                f"max joint step {avoid_step:.3f}rad)."
            )
        else:
            avoidance_status = "rejected_large_joint_step" if not continuous_enough else "rejected"
            print(
                f"[IK] Keeping smooth candidate: collision-aware candidate had "
                f"{avoid_collision_frames} collision frames and max joint step "
                f"{avoid_step:.3f}rad "
                f"(limit {COLLISION_AVOIDANCE_MAX_JOINT_STEP:.3f}rad)."
            )

    capture_gate_valid = np.asarray(capture_metrics["capture_valid"], dtype=bool)
    collision_free = np.asarray(collision_metrics["collision_free"], dtype=bool)
    capture_valid = capture_gate_valid
    flags = [bool(ok) for ok in capture_valid]
    traj_info["capture_gate_valid"] = capture_gate_valid
    traj_info["capture_valid"] = capture_valid
    traj_info["collision_free"] = collision_free
    traj_info["collision_avoidance_status"] = avoidance_status
    traj_info["capture_pos_err"] = capture_metrics["pos_err"]
    traj_info["capture_look_deg"] = capture_metrics["look_deg"]
    traj_info["capture_max_pos_err"] = CAPTURE_MAX_POS_ERR
    traj_info["capture_max_look_deg"] = CAPTURE_MAX_LOOK_DEG
    traj_info["collision_count"] = collision_metrics["collision_count"]
    traj_info["collision_min_dist"] = collision_metrics["min_contact_dist"]
    traj_info["collision_contacts"] = collision_metrics["contacts"]
    traj_info["target_rotation_err_deg"] = target_rot_deg

    n_ok = int(np.count_nonzero(capture_valid))
    n_fail = len(capture_valid) - n_ok
    print(
        f"\n[IK] Capture-valid frames: {n_ok}/{len(capture_valid)} "
        f"(look <= {CAPTURE_MAX_LOOK_DEG:.1f}deg, pos <= {CAPTURE_MAX_POS_ERR*1000:.1f}mm)"
    )
    n_collision = int(np.count_nonzero(~collision_free))
    print(f"[IK] Collision-free frames: {len(collision_free) - n_collision}/{len(collision_free)}")
    if n_collision:
        first = int(np.flatnonzero(~collision_free)[0])
        contacts = collision_metrics["contacts"][first]
        detail = ""
        if contacts:
            detail = f" first={contacts[0]['geom1']} <-> {contacts[0]['geom2']}"
        print(f"     First collision waypoint: {first + 1}{detail}")
    print(
        f"[IK] Camera target rotation error: "
        f"median={np.median(target_rot_deg):.2f}deg, "
        f"p95={np.percentile(target_rot_deg, 95):.2f}deg, "
        f"max={np.max(target_rot_deg):.2f}deg"
    )

    if n_fail > 0:
        failed = [i + 1 for i, ok in enumerate(capture_valid) if not ok]
        print(f"     Non-capture waypoints: {failed}")
        print("     These frames are still rendered; only segment boundaries are saved as black separators.")

    original_duration = float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0
    time_values = retime_joint_trajectory(time_values, Q, max_joint_speed=MAX_JOINT_SPEED)
    retimed_duration = float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0
    if retimed_duration > original_duration + 1e-6:
        print(
            f"[IK] Retimed trajectory for max joint speed {MAX_JOINT_SPEED:.2f} rad/s: "
            f"{original_duration:.2f}s -> {retimed_duration:.2f}s"
        )

    return Q, flags, time_values, angles, camera_positions, camera_rotations, camera_poses, tcp_poses_world, traj_info


def compute_ik_weld_trajectory(
    model, data,
    ply_path=None,
    verbose=True,
    retries=DEFAULT_IK_RETRIES,
    rng_seed=7,
    standoff=None,
    status_cb=None,
    progress_cb=None,
):
    """Compute joint trajectory for the welding path.

    Seam geometry is extracted from the reconstructed PLY, then weld EE poses
    are generated and solved with IK.  The torch Z axis (+Z of EE) points toward
    the weld seam, so axis_sign=+1.0 (opposite of the camera inspection case).
    """
    print("\n[IK-WELD] 용접 경로 IK 계산 중...")

    if ply_path is None:
        ply_path = WELD_PLY_PATH
    if not os.path.exists(ply_path):
        raise FileNotFoundError(
            f"[IK-WELD] PLY 파일을 찾을 수 없습니다: {ply_path}\n"
            "먼저 mujoco_viewer.py --ik --camera --export-csv --no-viewer 와 "
            "Reconstruct_3D.py 를 실행하세요."
        )

    from welding.seam_extraction import extract_seam
    from welding.weld_trajectory import generate_weld_poses
    from welding.weld_trajectory import WELD_STANDOFF_M as _STANDOFF
    from welding.weld_trajectory import WELD_TOOL_LENGTH_M as _TOOL_LEN

    seam = extract_seam(ply_path, verbose=verbose)

    # PLY는 캡처 당시의 파이프 위치 기준. scene의 현재 파이프 위치가 다르면
    # seam을 그 차이만큼 평행이동해 정합한다. 캡처 위치는 사이드카 JSON이 있으면
    # 그 값을(UI에서 재스캔한 경우), 없으면 기본 상수를 사용.
    capture_pipe_pos = WELD_PLY_CAPTURE_PIPE_POS
    _sidecar = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(ply_path)),
                                            "..", "capture_pipe_pos.json"))
    if os.path.isfile(_sidecar):
        import json as _json
        with open(_sidecar) as _f:
            capture_pipe_pos = np.asarray(_json.load(_f)["pipe_pos"], dtype=float)
    pipe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pipe_flange_assembly")
    if pipe_bid >= 0:
        seam_shift = model.body_pos[pipe_bid] - capture_pipe_pos
        if np.linalg.norm(seam_shift) > 1e-9:
            seam["center"] = np.asarray(seam["center"], dtype=float) + seam_shift
            print(f"[IK-WELD] seam을 현재 파이프 위치에 정합: shift={np.round(seam_shift, 4)}")

    _eff_standoff = _STANDOFF if standoff is None else float(standoff)
    weld_poses_full, angles_full = generate_weld_poses(
        center=seam["center"],
        radius=seam["radius"],
        normal=seam["normal"],
        exclude_bottom=False,
        standoff=_eff_standoff,
    )

    _reach           = _eff_standoff + _TOOL_LEN
    _torch_tip_in_ee = np.array([0.01384, -0.00829, -0.04733])
    seam_pts_full    = [T[:3, 3] + _reach * T[:3, 2] for T in weld_poses_full]

    _tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    _ee_bid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee")
    from control.franka_ik_solver import joint_limits, ARM_JOINT_NAMES
    _jlimits = joint_limits(model, mujoco)
    _jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm) for nm in ARM_JOINT_NAMES]
    _dof_idx = np.array([model.jnt_dofadr[i] for i in _jnt_ids], dtype=int)

    weld_posture_bias = dict(WELD_ABOVE_POSTURE_BIAS, q_ref=WELD_BIASED_READY, q_weight=0.05)
    _ik_base = dict(
        retries=retries,
        rng_seed=rng_seed,
        verbose=verbose,
        axis_col=WELD_LOOK_AXIS_COL,
        axis_sign=WELD_LOOK_AXIS_SIGN,
        posture_bias=weld_posture_bias,
        continuity_weight=0.18,
        max_capture_look_deg=15.0,
        max_capture_pos_err=0.005,
        invalid_candidate_penalty=0.0,
        collision_penalty=COLLISION_AVOIDANCE_DEFAULT_PENALTY,
        collision_margin=0.0,
        target_rotation_weight=0.3,
        pose_rotation_scale=np.deg2rad(90.0),
        site_name=TCP_SITE_NAME,
    )

    def _newton_correct(Q1_raw, seg_seam):
        """Pass-2: Newton IK correction so torch_tip lands on seam."""
        Q1s = np.array(Q1_raw, dtype=float)
        out = []
        for q0, sp in zip(Q1s, seg_seam):
            q = q0.copy()
            for _ in range(80):
                set_arm_qpos(model, data, mujoco, q)
                mujoco.mj_forward(model, data)
                R_ee = data.xmat[_ee_bid].reshape(3, 3)
                target_tcp = sp - R_ee @ _torch_tip_in_ee
                err = target_tcp - data.site_xpos[_tcp_sid]
                if np.linalg.norm(err) < 1e-4:
                    break
                J = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, J, None, _tcp_sid)
                J_arm = J[:, _dof_idx]
                dq = np.linalg.lstsq(
                    J_arm.T @ J_arm + 1e-6 * np.eye(NDOF),
                    J_arm.T @ err, rcond=None,
                )[0]
                q = np.clip(q + dq, _jlimits[:, 0], _jlimits[:, 1])
            out.append(q)
        return out

    def _solve_segment(seg_poses, seg_seam, seg_angles, name, q_start,
                       wp_offset=0, wp_total=0):
        sn = len(seg_poses)
        p1 = []
        for sp, T in zip(seg_seam, seg_poses):
            T2 = T.copy()
            T2[:3, 3] = sp - T[:3, :3] @ _torch_tip_in_ee
            p1.append(T2)
        print(f"[IK-WELD] {name}: Pass-1 ({sn} wp)...")
        _seg_prog = None
        if progress_cb is not None and wp_total > 0:
            def _seg_prog(i, n):
                progress_cb(wp_offset + i, wp_total)
        Q1, _ = solve_trajectory(
            model, data, mujoco, p1,
            q_start=np.asarray(q_start, dtype=float).copy(),
            look_target=seg_seam,
            posture_weights=np.ones(sn),
            progress_cb=_seg_prog,
            **_ik_base,
        )
        Q_seg = _newton_correct(Q1, seg_seam)
        print(f"[IK-WELD] {name}: done ({sn} wp)")
        return Q_seg

    # ── 4개 segment: 12시→6시(CW, 3시 경유) → 12시(CCW, 3시 경유)
    #                → 6시(CCW, 9시 경유) → 12시(CW, 9시 경유)
    # phi: 0°=3시, 90°=12시, 180°=9시, 270°=6시 (phi 증가 = 반시계)
    # 모든 segment 경계가 12시/6시 동일 지점에서 만나 위치 불연속이 없다.
    adeg = np.degrees(angles_full)

    def _range_idx(phi_lo, phi_hi, rev):
        idx = np.where((adeg >= phi_lo) & (adeg <= phi_hi))[0]
        return idx[::-1] if rev else idx

    _seg_down_3 = np.concatenate([_range_idx(0, 90, True),
                                  _range_idx(270, 360, True)])   # 12시→6시 (3시 경유)
    _seg_down_9 = _range_idx(90, 270, False)                     # 12시→6시 (9시 경유)

    # 복귀 패스는 같은 pose를 역순으로 지나므로 (push angle이 방향 무관하게 동일)
    # 하강 패스의 IK 해를 미러링해 재사용 — 왕복 간 브랜치 플립을 원천 차단.
    seg_defs = [
        (_seg_down_3,        "12시→6시 (CW, 3시)",  None, None),
        (_seg_down_3[::-1],  "6시→12시 (CCW, 3시)", 0,    None),
        (_seg_down_9,        "12시→6시 (CCW, 9시)", None, WELD_LEFT_HALF_SEED),
        (_seg_down_9[::-1],  "6시→12시 (CW, 9시)",  2,    None),
    ]

    Q_all, angles_all, weld_poses_all, seam_pts_all = [], [], [], []
    seg_solutions: list[list] = []
    total_solve_wp = sum(len(idx) for idx, _, mirror_of, _ in seg_defs if mirror_of is None)
    solved_wp = 0
    for seg_i, (idx, name, mirror_of, seed_override) in enumerate(seg_defs):
        if len(idx) == 0:
            seg_solutions.append([])
            continue
        if status_cb is not None:
            status_cb(name, seg_i + 1, len(seg_defs))
        sp   = [seam_pts_full[i]  for i in idx]
        wp   = [weld_poses_full[i] for i in idx]
        ang  = angles_full[idx]
        if mirror_of is not None:
            Q_seg = list(seg_solutions[mirror_of])[::-1]
            print(f"[IK-WELD] {name}: 미러 재사용 ({len(Q_seg)} wp)")
        else:
            # 시드 우선순위: segment 전용 시드 > 이전 segment 마지막 해(warm-start) > 기본
            if seed_override is not None:
                q_seed = seed_override
            else:
                q_seed = Q_all[-1] if Q_all else WELD_BIASED_READY
            Q_seg = _solve_segment(wp, sp, ang, name, q_seed,
                                   wp_offset=solved_wp, wp_total=total_solve_wp)
            solved_wp += len(idx)
        seg_solutions.append(Q_seg)
        Q_all.extend(Q_seg)
        angles_all.extend(ang.tolist())
        weld_poses_all.extend(wp)
        seam_pts_all.extend(sp)

    Q          = Q_all
    angles     = np.array(angles_all)
    weld_poses = weld_poses_all
    seam_pts   = seam_pts_all
    flags      = [True] * len(Q)
    positions  = np.array([sp - T[:3, :3] @ _torch_tip_in_ee
                           for sp, T in zip(seam_pts, weld_poses)])
    n = len(Q)

    time_values = np.linspace(0.0, (n - 1) * TRAJECTORY_DT, n)
    orig_dur = float(time_values[-1] - time_values[0]) if n > 1 else 0.0
    time_values = retime_joint_trajectory(time_values, Q, max_joint_speed=MAX_JOINT_SPEED)
    new_dur = float(time_values[-1] - time_values[0]) if n > 1 else 0.0
    if new_dur > orig_dur + 1e-6:
        print(f"[IK-WELD] 속도 제한 재타이밍: {orig_dur:.2f}s → {new_dur:.2f}s")

    n_ok = int(np.count_nonzero(flags))
    print(f"[IK-WELD] IK 성공 (필터 후): {n_ok}/{n}")

    # ── 충돌 진단 ────────────────────────────────────────────────────────────
    col_metrics = evaluate_collision_trajectory(model, data, mujoco, Q, collision_margin=0.0)
    col_counts = col_metrics["collision_count"]
    col_dists  = col_metrics["min_contact_dist"]
    col_pairs  = col_metrics["contacts"]
    n_col = int(np.sum(col_counts > 0))
    print(f"[IK-WELD] 충돌 waypoints: {n_col}/{n}", end="")
    if n_col > 0:
        finite_dists = col_dists[np.isfinite(col_dists)]
        worst_dist = float(np.min(finite_dists)) if len(finite_dists) else float("inf")
        print(f"  (최대 관통 깊이: {-worst_dist*1000:.1f} mm)")
        print("[IK-WELD] 충돌 waypoint 상세 (phi, 링크):")
        for i, (cnt, pairs) in enumerate(zip(col_counts, col_pairs)):
            if cnt > 0:
                phi_deg = float(np.degrees(angles[i]))
                pair_str = ", ".join(f"{p['geom1']}↔{p['geom2']}" for p in pairs[:2])
                print(f"  WP{i+1:3d} phi={phi_deg:6.1f}°  {pair_str}")
    else:
        print("  (충돌 없음)")

    return Q, flags, time_values, angles, positions, weld_poses, seam


# ── FK verification ──────────────────────────────────────────────────────────

def verify_ik(Q, tcp_poses):
    """Print camera-center site position error for each waypoint after IK."""
    print("\n[FK] IK verification:")
    max_tcp_err = 0.0
    for i, (q, T_des) in enumerate(zip(Q, tcp_poses)):
        set_arm_qpos(verify_ik.model, verify_ik.data, mujoco, q)
        T_act = site_pose(verify_ik.model, verify_ik.data, mujoco, CAMERA_SITE_NAME)
        err = np.linalg.norm(T_des[:3,3] - T_act[:3,3])
        max_tcp_err = max(max_tcp_err, err)
        if err > 1e-3:
            print(f"  [WARN] WP {i+1:2d}: camera_pos_err = {err*1000:.2f} mm")
    print(f"  Max camera position error: {max_tcp_err*1000:.3f} mm")


def print_joint_log(Q, flags, angles, traj_info=None, top_n=20, full=False, label="raw IK"):
    """Print joint-space discontinuities so branch switches are easy to spot."""
    Q = np.asarray(Q, dtype=float)
    if len(Q) < 2:
        print("[JOINT] Not enough waypoints for joint jump analysis.")
        return

    dq = np.diff(Q, axis=0)
    dq_norm = np.linalg.norm(dq, axis=1)
    dq_abs_max = np.max(np.abs(dq), axis=1)
    if angles is None:
        angles_deg = np.full(len(Q), np.nan, dtype=float)
    else:
        angles_deg = np.rad2deg(np.asarray(angles, dtype=float))
        if angles_deg.shape != (len(Q),):
            angles_deg = np.full(len(Q), np.nan, dtype=float)

    flags = np.asarray(flags, dtype=bool) if flags is not None else np.ones(len(Q), dtype=bool)
    if flags.shape != (len(Q),):
        flags = np.ones(len(Q), dtype=bool)

    if traj_info is not None and "segment_id" in traj_info:
        segments = np.asarray(traj_info["segment_id"], dtype=int)
        if segments.shape != (len(Q),):
            segments = np.zeros(len(Q), dtype=int)
    else:
        segments = np.zeros(len(Q), dtype=int)

    def angle_text(value):
        return "    n/a" if np.isnan(value) else f"{value:7.1f}"

    print(f"\n[JOINT] Joint trajectory jump audit ({label})")
    print(f"  capture valid: {int(np.count_nonzero(flags))}/{len(flags)}")
    print(f"  max ||dq||: {float(np.max(dq_norm)):.3f} rad")
    print(f"  max |dq_i|: {float(np.max(dq_abs_max)):.3f} rad")
    print(f"  mean ||dq||: {float(np.mean(dq_norm)):.3f} rad")

    print("\n[JOINT] Top joint jumps")
    print("rank  edge    seg  phi_from->to      ||dq||   max|dq_i|   dq[1..6]")
    for rank, edge in enumerate(np.argsort(dq_norm)[-top_n:][::-1], start=1):
        print(
            f"{rank:>4}  {edge+1:03d}->{edge+2:03d}  "
            f"{int(segments[edge])}->{int(segments[edge+1])}  "
            f"{angle_text(angles_deg[edge])}->{angle_text(angles_deg[edge+1])}  "
            f"{dq_norm[edge]:8.3f}  {dq_abs_max[edge]:9.3f}  "
            f"{np.array2string(dq[edge], precision=3, suppress_small=True)}"
        )

    if not full:
        return

    print("\n[JOINT] Full compact joint log")
    print("idx seg phi_deg ok q1 q2 q3 q4 q5 q6 |dq|")
    print("000 --      -- -- " + " ".join(f"{v: .3f}" for v in Q[0]) + "  0.000")
    for i in range(1, len(Q)):
        marker = "JUMP" if dq_norm[i - 1] > 0.8 or dq_abs_max[i - 1] > 0.6 else ""
        print(
            f"{i:03d} {int(segments[i]):>2d} {angle_text(angles_deg[i])} {int(flags[i])} "
            + " ".join(f"{v: .3f}" for v in Q[i])
            + f" {dq_norm[i - 1]:7.3f} {marker}"
        )


def print_collision_log(angles, traj_info=None, top_n=20):
    """Print robot-involved collision contacts recorded for the IK trajectory."""
    if not traj_info or "collision_count" not in traj_info:
        print("[COLLISION] No collision diagnostics available.")
        return

    counts = np.asarray(traj_info["collision_count"], dtype=int)
    contacts = traj_info.get("collision_contacts", [[] for _ in range(len(counts))])
    if angles is None:
        angles_deg = np.full(len(counts), np.nan, dtype=float)
    else:
        angles_deg = np.rad2deg(np.asarray(angles, dtype=float))
        if angles_deg.shape != counts.shape:
            angles_deg = np.full(len(counts), np.nan, dtype=float)

    bad = np.flatnonzero(counts > 0)
    print(f"\n[COLLISION] collision-free: {len(counts) - len(bad)}/{len(counts)}")
    if len(bad) == 0:
        return

    def angle_text(value):
        return "n/a" if np.isnan(value) else f"{value:.1f}"

    print("[COLLISION] First contact rows")
    print("idx  phi_deg  n  first_contact")
    for idx in bad[:max(1, int(top_n))]:
        pair = ""
        if contacts[idx]:
            first = contacts[idx][0]
            pair = f"{first['geom1']} <-> {first['geom2']} ({first['dist']*1000:.1f}mm)"
        print(f"{idx:03d}  {angle_text(angles_deg[idx]):>7}  {counts[idx]:>2d}  {pair}")


def save_trajectory_csv(path, Q, flags, angles, camera_poses, tcp_poses, traj_info=None):
    """Save desired EE poses and IK joint trajectory in one CSV file."""
    fieldnames = [
        "index", "angle_deg", "capture_valid", "capture_look_deg", "capture_pos_err_mm",
        "camera_x", "camera_y", "camera_z", "camera_qw", "camera_qx", "camera_qy", "camera_qz",
        "tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz",
        "q1", "q2", "q3", "q4", "q5", "q6",
    ]
    look_deg = None if traj_info is None else traj_info.get("capture_look_deg")
    pos_err = None if traj_info is None else traj_info.get("capture_pos_err")
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
                "capture_valid": int(ok),
                "capture_look_deg": "" if look_deg is None else float(look_deg[i]),
                "capture_pos_err_mm": "" if pos_err is None else float(pos_err[i]) * 1000.0,
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


def render_inspection_cameras(model, data, Q, traj_info=None, out_dir="inspection_frames", camera_name="d405_camera",
                              progress_cb=None):
    """Render D405-style metric depth maps from the selected fixed camera.

    progress_cb: optional callable(i, n) invoked after each rendered waypoint;
    returning False aborts the capture (used by the control UI / E-STOP).
    """
    if not MUJOCO_AVAILABLE:
        return
    os.makedirs(out_dir, exist_ok=True)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        print(f"[WARN] camera '{camera_name}' not found in model.")
        return False

    set_d405_depth_rendering(model, cam_id)
    renderer = mujoco.Renderer(model, height=D405_DEPTH_HEIGHT, width=D405_DEPTH_WIDTH)
    renderer.enable_depth_rendering()

    depth_dir = os.path.join(out_dir, "depth_meters")
    png_dir = os.path.join(out_dir, "depth_png")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    segment_ids = None
    separator_count = 0
    capture_valid = np.ones(len(Q), dtype=bool)
    capture_look_deg = np.full(len(Q), np.nan, dtype=float)
    capture_pos_err = np.full(len(Q), np.nan, dtype=float)
    if traj_info is not None:
        segment_ids = traj_info.get("segment_id")
        if segment_ids is not None:
            segment_ids = np.asarray(segment_ids)
            separator_count = int(np.count_nonzero(segment_ids[1:] != segment_ids[:-1]))
        if "capture_valid" in traj_info:
            capture_valid = np.asarray(traj_info["capture_valid"], dtype=bool)
        if "capture_look_deg" in traj_info:
            capture_look_deg = np.asarray(traj_info["capture_look_deg"], dtype=float)
        if "capture_pos_err" in traj_info:
            capture_pos_err = np.asarray(traj_info["capture_pos_err"], dtype=float)

    total_frames = len(Q) + separator_count
    invalid_count = int(len(Q) - np.count_nonzero(capture_valid))
    print(
        f"\n[CAM] Rendering {len(Q)} D405 depth frames from '{camera_name}' "
        f"to '{out_dir}/'..."
    )
    if separator_count:
        print(f"[CAM] Inserting {separator_count} black separator frames at segment boundaries.")
    if invalid_count:
        print(f"[CAM] Saving {invalid_count} non-capture look-at frames as normal rendered frames.")

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
        if not bool(capture_valid[i]):
            print(
                f"[CAM] frame_{output_i:03d}: rendered non-capture wp {i+1} "
                f"(look={capture_look_deg[i]:.1f}deg, pos={capture_pos_err[i]*1000.0:.1f}mm)"
            )
        output_i += 1
        if progress_cb is not None and progress_cb(i + 1, len(Q)) is False:
            renderer.close()
            print("[CAM] Capture aborted by progress callback (E-STOP).")
            return False

    renderer.close()
    print(f"[CAM] Saved metric depth arrays: {depth_dir}/frame_*.npy")
    print(f"[CAM] Saved depth visualizations: {png_dir}/frame_*.png")
    if separator_count or invalid_count:
        valid_capture_count = int(np.count_nonzero(capture_valid))
        print(
            f"[CAM] Frame count: {valid_capture_count} valid captures + "
            f"{invalid_count} non-capture rendered frames + "
            f"{separator_count} black separators = {total_frames} files"
        )
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
    return True


# ── Interactive viewer ───────────────────────────────────────────────────────

def _pchip_joint_tangent(time_values, Q, index):
    """Shape-preserving joint velocity estimate for smooth viewer playback."""
    n = len(Q)
    if n < 2:
        return np.zeros(Q.shape[1], dtype=float)

    if index <= 0:
        h = max(float(time_values[1] - time_values[0]), 1e-12)
        return (Q[1] - Q[0]) / h
    if index >= n - 1:
        h = max(float(time_values[-1] - time_values[-2]), 1e-12)
        return (Q[-1] - Q[-2]) / h

    h_prev = max(float(time_values[index] - time_values[index - 1]), 1e-12)
    h_next = max(float(time_values[index + 1] - time_values[index]), 1e-12)
    d_prev = (Q[index] - Q[index - 1]) / h_prev
    d_next = (Q[index + 1] - Q[index]) / h_next

    tangent = np.zeros_like(d_prev)
    same_direction = d_prev * d_next > 0.0
    if np.any(same_direction):
        w_prev = 2.0 * h_next + h_prev
        w_next = h_next + 2.0 * h_prev
        tangent[same_direction] = (w_prev + w_next) / (
            w_prev / d_prev[same_direction] + w_next / d_next[same_direction]
        )
    return tangent


def interpolate_q(time_values, Q, t, smooth=SMOOTH_PLAYBACK_INTERPOLATION):
    if t <= time_values[0]:
        return Q[0]
    if t >= time_values[-1]:
        return Q[-1]
    hi = int(np.searchsorted(time_values, t, side="right"))
    lo = hi - 1
    span = max(time_values[hi] - time_values[lo], 1e-12)
    alpha = (t - time_values[lo]) / span
    if not smooth or len(Q) < 3:
        return (1.0 - alpha) * Q[lo] + alpha * Q[hi]

    m0 = _pchip_joint_tangent(time_values, Q, lo)
    m1 = _pchip_joint_tangent(time_values, Q, hi)
    a2 = alpha * alpha
    a3 = a2 * alpha
    h00 = 2.0 * a3 - 3.0 * a2 + 1.0
    h10 = a3 - 2.0 * a2 + alpha
    h01 = -2.0 * a3 + 3.0 * a2
    h11 = a3 - a2
    return h00 * Q[lo] + h10 * span * m0 + h01 * Q[hi] + h11 * span * m1


def prepend_zero_to_start(
    model,
    time_values,
    Q,
    approach_duration=APPROACH_DURATION,
    via_q=RIGHT_BIASED_READY,
    via_fraction=0.45,
):
    limits = joint_limits(model, mujoco)
    q_zero = np.clip(RIGHT_BIASED_READY.copy(), limits[:, 0], limits[:, 1])
    q_start = Q[0]

    if approach_duration <= 0.0:
        return time_values, Q

    nominal_dt = float(np.median(np.diff(time_values))) if len(time_values) > 1 else 0.02
    nominal_dt = max(nominal_dt, 0.01)
    n_steps = max(2, int(np.ceil(approach_duration / nominal_dt)))

    if via_q is None:
        approach_time = np.linspace(0.0, approach_duration, n_steps + 1)
        u = approach_time / approach_duration
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        approach_q = q_zero[None, :] + s[:, None] * (q_start - q_zero)[None, :]
    else:
        q_via = np.clip(np.asarray(via_q, dtype=float), limits[:, 0], limits[:, 1])
        via_fraction = float(np.clip(via_fraction, 0.1, 0.9))
        n_to_via = max(2, int(round(n_steps * via_fraction)))
        n_to_start = max(2, n_steps - n_to_via)
        t_via = approach_duration * via_fraction

        t1 = np.linspace(0.0, t_via, n_to_via + 1)
        u1 = t1 / max(t_via, 1e-9)
        s1 = 10.0 * u1**3 - 15.0 * u1**4 + 6.0 * u1**5
        q1 = q_zero[None, :] + s1[:, None] * (q_via - q_zero)[None, :]

        t2 = np.linspace(t_via, approach_duration, n_to_start + 1)[1:]
        u2 = (t2 - t_via) / max(approach_duration - t_via, 1e-9)
        s2 = 10.0 * u2**3 - 15.0 * u2**4 + 6.0 * u2**5
        q2 = q_via[None, :] + s2[:, None] * (q_start - q_via)[None, :]

        approach_time = np.concatenate([t1, t2])
        approach_q = np.vstack([q1, q2])

    shifted_time = approach_duration + (time_values - time_values[0])
    return (
        np.concatenate([approach_time, shifted_time[1:]]),
        np.vstack([approach_q, Q[1:]]),
    )


def _smooth_joint_bridge(q0, q1, steps=PLAYBACK_BRIDGE_STEPS):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if np.linalg.norm(q1 - q0) < 1e-9:
        return np.empty((0, len(q0)), dtype=float)
    steps = max(2, int(steps))
    u = np.linspace(0.0, 1.0, steps + 1)[1:]
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    return q0[None, :] + s[:, None] * (q1 - q0)[None, :]


def _densify_joint_path(Q, max_sample_step=PLAYBACK_MAX_SAMPLE_STEP):
    """Resample the path with small constant joint increments for continuous playback."""
    Q = np.asarray(Q, dtype=float)
    if len(Q) < 2:
        return Q.copy()

    dense = [Q[0].copy()]
    for q0, q1 in zip(Q[:-1], Q[1:]):
        max_delta = float(np.max(np.abs(q1 - q0)))
        steps = max(1, int(np.ceil(max_delta / max(float(max_sample_step), 1e-6))))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            dense.append((1.0 - alpha) * q0 + alpha * q1)
    return np.asarray(dense, dtype=float)


def _retime_playback_path(playback_q):
    playback_time = np.arange(len(playback_q), dtype=float) * PLAYBACK_NOMINAL_DT
    nominal_duration = float(playback_time[-1] - playback_time[0]) if len(playback_time) > 1 else 0.0
    playback_time = retime_joint_trajectory(
        playback_time,
        playback_q,
        max_joint_speed=MAX_JOINT_SPEED,
        max_joint_accel=PLAYBACK_MAX_JOINT_ACCEL,
        max_joint_jerk=PLAYBACK_MAX_JOINT_JERK,
    )
    metrics = joint_motion_metrics(playback_time, playback_q)
    print(
        f"[VIEWER] Playback retime: {nominal_duration:.2f}s -> {metrics['duration']:.2f}s "
        f"(max speed={metrics['max_speed']:.2f}rad/s, "
        f"accel={metrics['max_accel']:.2f}rad/s^2, "
        f"jerk={metrics['max_jerk']:.2f}rad/s^3)"
    )
    return playback_time


def _compute_cartesian_playback_path(
    model,
    data,
    q_start,
    max_waypoints=PLAYBACK_CARTESIAN_WAYPOINTS,
    retries=PLAYBACK_CARTESIAN_RETRIES,
    rng_seed=17,
):
    """
    Build viewer playback samples from dense Cartesian camera targets.

    The raw IK path chooses a stable joint branch with the slower path planner.
    This pass then follows the same Cartesian circle more densely with
    sequential IK, so the viewer does not rely on long joint-space chords
    between sparse waypoints.
    """
    if max_waypoints is None or int(max_waypoints) <= 0:
        return None

    (
        time_values,
        _angles,
        _camera_positions,
        _camera_rotations,
        camera_poses,
        look_targets,
        dense_info,
    ) = generate_segmented_reference(
        max_waypoints=int(max_waypoints),
        return_targets=True,
        return_info=True,
    )
    tcp_poses = camera_poses_to_tcp_poses(camera_poses)
    posture_weights = np.zeros(len(tcp_poses), dtype=float)

    print(
        f"[VIEWER] Solving dense Cartesian playback IK: "
        f"{len(tcp_poses)} camera samples..."
    )
    playback_q, _ = solve_trajectory(
        model,
        data,
        mujoco,
        tcp_poses,
        look_target=look_targets,
        q_start=RIGHT_BIASED_READY.copy(),
        retries=max(0, int(retries)),
        rng_seed=rng_seed,
        verbose=False,
        axis_col=EE_LOOK_AXIS_COL,
        axis_sign=EE_LOOK_AXIS_SIGN,
        continuity_weight=1.0,
        limit_weight=0.04,
        posture_bias=None,
        posture_weights=posture_weights,
        max_joint_step=0.18,
        joint_step_weight=120.0,
        hard_max_joint_step=0.35,
        path_planning=False,
        max_capture_look_deg=CAPTURE_MAX_LOOK_DEG,
        max_capture_pos_err=CAPTURE_MAX_POS_ERR,
        invalid_candidate_penalty=1.0e6 if ENFORCE_CAPTURE_GATE_IN_IK else 0.0,
        collision_penalty=30.0,
        collision_margin=COLLISION_MARGIN,
        site_name=CAMERA_SITE_NAME,
    )

    capture_metrics = evaluate_look_at_trajectory(
        model,
        data,
        mujoco,
        playback_q,
        tcp_poses,
        look_targets,
        axis_col=EE_LOOK_AXIS_COL,
        axis_sign=EE_LOOK_AXIS_SIGN,
        site_name=CAMERA_SITE_NAME,
        max_pos_err=CAPTURE_MAX_POS_ERR,
        max_look_deg=CAPTURE_MAX_LOOK_DEG,
    )
    collision_metrics = evaluate_collision_trajectory(
        model,
        data,
        mujoco,
        playback_q,
        collision_margin=COLLISION_MARGIN,
    )
    capture_valid = np.asarray(capture_metrics["capture_valid"], dtype=bool)
    target_rot_deg = target_rotation_error_degrees(model, data, playback_q, tcp_poses)
    valid_count = int(np.count_nonzero(capture_valid))
    collision_count = int(np.count_nonzero(~collision_metrics["collision_free"]))
    max_step = max_adjacent_joint_step(playback_q)
    print(
        f"[VIEWER] Dense Cartesian playback valid: {valid_count}/{len(playback_q)} "
        f"(max pos={np.max(capture_metrics['pos_err']) * 1000.0:.2f}mm, "
        f"max look={np.max(capture_metrics['look_deg']):.2f}deg, "
        f"max rot={np.max(target_rot_deg):.2f}deg, "
        f"collisions={collision_count}, "
        f"max joint step={max_step:.3f}rad)"
    )

    original_duration = float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0
    playback_time = retime_joint_trajectory(
        time_values,
        playback_q,
        max_joint_speed=MAX_JOINT_SPEED,
        max_joint_accel=PLAYBACK_MAX_JOINT_ACCEL,
        max_joint_jerk=PLAYBACK_MAX_JOINT_JERK,
    )
    metrics = joint_motion_metrics(playback_time, playback_q)
    if metrics["duration"] > original_duration + 1e-6:
        print(
            f"[VIEWER] Dense Cartesian playback retime: "
            f"{original_duration:.2f}s -> {metrics['duration']:.2f}s "
            f"(max speed={metrics['max_speed']:.2f}rad/s, "
            f"accel={metrics['max_accel']:.2f}rad/s^2, "
            f"jerk={metrics['max_jerk']:.2f}rad/s^3)"
        )

    return playback_q, playback_time, {
        "groups": [np.arange(len(playback_q), dtype=int)],
        "bridge_samples": 0,
        "cartesian_playback": True,
        "capture_valid": capture_valid,
        "collision_count": collision_metrics["collision_count"],
        "max_joint_step": max_step,
    }


def _valid_capture_groups(capture_valid, segment_ids=None):
    capture_valid = np.asarray(capture_valid, dtype=bool)
    valid_idx = np.flatnonzero(capture_valid)
    if len(valid_idx) == 0:
        return []

    if segment_ids is None:
        segment_ids = np.zeros(len(capture_valid), dtype=int)
    else:
        segment_ids = np.asarray(segment_ids, dtype=int)

    groups = []
    start = int(valid_idx[0])
    prev = int(valid_idx[0])
    for idx in valid_idx[1:]:
        idx = int(idx)
        same_capture_run = idx == prev + 1
        same_segment = int(segment_ids[idx]) == int(segment_ids[prev])
        if not (same_capture_run and same_segment):
            groups.append(np.arange(start, prev + 1, dtype=int))
            start = idx
        prev = idx
    groups.append(np.arange(start, prev + 1, dtype=int))
    return groups


def build_capture_playback_trajectory(
    model,
    data,
    Q,
    time_values,
    traj_info=None,
    via_q=RIGHT_BIASED_READY,
    capture_only=PLAYBACK_CAPTURE_ONLY,
    playback_waypoints=PLAYBACK_CARTESIAN_WAYPOINTS,
    retries=PLAYBACK_CARTESIAN_RETRIES,
    rng_seed=17,
):
    """
    Build a viewer-only path that never follows invalid capture IK poses.

    Raw IK waypoints are still kept for diagnostics and frame export.  The
    interactive viewer, however, should not animate through frames where the
    camera is known not to look at the seam; those poses are reposition noise.
    """
    Q = np.asarray(Q, dtype=float)
    if len(Q) == 0:
        return Q, np.asarray([], dtype=float), {"groups": [], "bridge_samples": 0}

    if not capture_only:
        if data is not None and playback_waypoints and int(playback_waypoints) > 0:
            cartesian_playback = _compute_cartesian_playback_path(
                model,
                data,
                Q[0],
                max_waypoints=playback_waypoints,
                retries=retries,
                rng_seed=rng_seed,
            )
            if cartesian_playback is not None:
                playback_q, playback_time, playback_info = cartesian_playback
                capture_valid = np.asarray(
                    playback_info.get("capture_valid", np.ones(len(playback_q), dtype=bool)),
                    dtype=bool,
                )
                collision_count = np.asarray(
                    playback_info.get("collision_count", np.zeros(len(playback_q), dtype=int)),
                    dtype=int,
                )
                max_step = float(playback_info.get("max_joint_step", max_adjacent_joint_step(playback_q)))
                if (
                    np.all(capture_valid)
                    and not np.any(collision_count > 0)
                    and max_step <= COLLISION_AVOIDANCE_MAX_JOINT_STEP
                ):
                    return cartesian_playback
                print(
                    "[VIEWER] Dense Cartesian playback is not fully valid/collision-free/smooth; "
                    "falling back to collision-checked joint playback."
                )

        playback_q = _densify_joint_path(Q)
        playback_time = _retime_playback_path(playback_q)
        collision_metrics = (
            evaluate_collision_trajectory(
                model,
                data,
                mujoco,
                playback_q,
                collision_margin=COLLISION_MARGIN,
            )
            if data is not None
            else None
        )
        collision_count = 0 if collision_metrics is None else int(
            np.count_nonzero(~collision_metrics["collision_free"])
        )
        print(
            f"[VIEWER] Playback path uses joint-space fallback between sparse IK waypoints: "
            f"{len(Q)} IK waypoints -> {len(playback_q)} playback samples "
            f"(collisions={collision_count}). Cartesian tracking may be approximate."
        )
        return playback_q, playback_time, {
            "groups": [np.arange(len(Q), dtype=int)],
            "bridge_samples": 0,
            "cartesian_playback": False,
            "collision_count": None if collision_metrics is None else collision_metrics["collision_count"],
        }

    if traj_info is None or "capture_valid" not in traj_info:
        playback_time = _retime_playback_path(Q)
        return Q, playback_time, {"groups": [np.arange(len(Q), dtype=int)], "bridge_samples": 0}

    capture_valid = np.asarray(traj_info["capture_valid"], dtype=bool)
    segment_ids = traj_info.get("segment_id")
    groups = _valid_capture_groups(capture_valid, segment_ids)
    if not groups:
        print("[VIEWER] No valid capture waypoint found; holding the first IK pose.")
        return Q[:1], np.asarray([0.0]), {"groups": [], "bridge_samples": 0}

    limits = joint_limits(model, mujoco)
    via_q = np.clip(np.asarray(via_q, dtype=float), limits[:, 0], limits[:, 1])

    playback_q = []
    source_index = []
    bridge_samples = 0

    def append_rows(rows, source):
        nonlocal bridge_samples
        rows = np.asarray(rows, dtype=float)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        for row in rows:
            if playback_q and np.linalg.norm(row - playback_q[-1]) < 1e-9:
                continue
            playback_q.append(row.copy())
            source_index.append(int(source))
            if source < 0:
                bridge_samples += 1

    for group_i, group in enumerate(groups):
        if group_i > 0:
            append_rows(_smooth_joint_bridge(playback_q[-1], via_q), -1)
            append_rows(_smooth_joint_bridge(via_q, Q[group[0]]), -1)
        for idx in group:
            append_rows(Q[int(idx)], int(idx))

    playback_q = _densify_joint_path(np.asarray(playback_q, dtype=float))
    playback_time = _retime_playback_path(playback_q)

    print(
        f"[VIEWER] Playback path uses {int(np.count_nonzero(capture_valid))} valid capture "
        f"waypoints in {len(groups)} groups; skipped {len(Q) - int(np.count_nonzero(capture_valid))} "
        f"invalid IK poses and inserted {bridge_samples} smooth reposition samples."
    )

    return playback_q, playback_time, {
        "groups": groups,
        "source_index": np.asarray(source_index, dtype=int),
        "bridge_samples": bridge_samples,
    }


def run_interactive(
    model,
    data,
    Q=None,
    positions=None,
    time_values=None,
    approach_duration=APPROACH_DURATION,
    fixed_camera_name=None,
    show_frames=True,
    overlay_poses=None,
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
    print("  Press [R]     to restart playback from the beginning.")
    print("  Press [Esc]   to exit.\n")
    if show_frames:
        print("[VIEWER] Showing World/Base and Pipe-Flange frames.")
    if Q is not None:
        mode = "smooth PCHIP" if SMOOTH_PLAYBACK_INTERPOLATION else "linear"
        print(f"[VIEWER] Joint playback interpolation: {mode}")

    paused = [False]
    restart_requested = [False]

    def key_callback(keycode):
        if keycode == ord(" "):
            paused[0] = not paused[0]
        elif keycode in (ord("R"), ord("r")):
            restart_requested[0] = True
            paused[0] = False

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

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.opt.sitegroup[:] = 1
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

        start_time = [time.time()]
        while viewer.is_running():
            now = time.time()

            if restart_requested[0]:
                restart_requested[0] = False
                start_time[0] = now
                if Q is not None and len(Q) > 0:
                    set_arm_qpos(model, data, mujoco, Q[0])
                else:
                    mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)

            if Q is not None and not paused[0]:
                traj_t = min((now - start_time[0]) * ANIMATION_SPEED, time_values[-1])
                q = interpolate_q(time_values, Q, traj_t)
                set_arm_qpos(model, data, mujoco, q)

            if show_frames:
                if overlay_poses is not None:
                    draw_weld_trajectory_overlay(viewer, overlay_poses)
                else:
                    draw_inspection_frames(viewer, model, data, show_tcp=True)

            viewer.sync()
            time.sleep(0.002)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Flange Inspection Viewer")
    parser.add_argument("--scene", default=SCENE_XML, help="MuJoCo scene XML path")
    parser.add_argument("--ik",         action="store_true", help="Compute IK and animate")
    parser.add_argument("--camera",     action="store_true", help="Render frames from the selected fixed camera")
    parser.add_argument("--camera-viewer", action="store_true", help="Open the interactive viewer through the selected fixed camera")
    parser.add_argument("--camera-name", default="d405_camera", help="MuJoCo fixed camera name to render.")
    parser.add_argument("--save-video", action="store_true", help="Save animation as video")
    parser.add_argument("--waypoints",  action="store_true", help="Print waypoints and exit")
    parser.add_argument("--verify",     action="store_true", help="Print IK FK verification")
    parser.add_argument("--joint-log",  action="store_true", help="Print largest joint-space jumps after IK")
    parser.add_argument("--joint-log-full", action="store_true", help="Print the full compact joint log after IK")
    parser.add_argument("--collision-log", action="store_true", help="Print robot collision contacts after IK")
    parser.add_argument("--no-viewer",  action="store_true", help="Do not open the interactive viewer")
    parser.add_argument("--retries",     type=int, default=DEFAULT_IK_RETRIES, help="Random restarts per waypoint for IK fallback")
    parser.add_argument("--rng-seed",    type=int, default=7, help="Random seed for deterministic IK retry fallback")
    parser.add_argument("--ik-waypoints", type=int, default=IK_WAYPOINTS,
                        help="Raw reference waypoints for IK solving")
    parser.add_argument("--playback-waypoints", type=int, default=PLAYBACK_CARTESIAN_WAYPOINTS,
                        help="Dense Cartesian IK samples used only for interactive viewer playback.")
    parser.add_argument("--approach-duration", type=float, default=APPROACH_DURATION,
                        help="Seconds to move from zero pose to the 12 o'clock start")
    parser.add_argument("--export-csv", nargs="?", const="inspection_frames/metadata.csv",
                        help="Write desired poses and joint trajectory to CSV")
    parser.add_argument("--hide-frames", action="store_true",
                        help="Do not draw World/Base and Pipe-Flange frame overlays")
    parser.add_argument("--show-trajectory", action="store_true",
                        help="Show reference trajectory circle markers in the viewer")
    parser.add_argument("--quiet-ik", action="store_true", help="Suppress per-waypoint IK logs")
    parser.add_argument("--weld", action="store_true",
                        help="용접 경로 IK 계산 및 MuJoCo 시뮬레이션 (seam 추출 → weld IK → viewer)")
    parser.add_argument("--collision-avoidance", action="store_true",
                        help="Run a slower second collision-aware IK candidate solve")
    parser.add_argument("--no-collision-avoidance", action="store_true",
                        help="Skip the second collision-aware IK candidate solve")
    args = parser.parse_args()
    global COLLISION_AVOIDANCE_PENALTY
    global SHOW_TRAJECTORY_MARKERS
    SHOW_TRAJECTORY_MARKERS = bool(args.show_trajectory)
    if args.collision_avoidance and not args.no_collision_avoidance:
        COLLISION_AVOIDANCE_PENALTY = COLLISION_AVOIDANCE_DEFAULT_PENALTY
    elif args.no_collision_avoidance:
        COLLISION_AVOIDANCE_PENALTY = 0.0

    # ── Waypoints only ────────────────────────────────────────────────────
    if args.waypoints:
        _, angles, positions, rotations, poses = generate_segmented_reference(max_waypoints=36)
        print_waypoints(angles, positions, rotations)
        return

    # ── Load model ───────────────────────────────────────────────────────
    scene_xml = os.path.abspath(args.scene)
    if not os.path.exists(scene_xml):
        print(f"[ERROR] scene XML not found at: {scene_xml}")
        sys.exit(1)

    if not MUJOCO_AVAILABLE:
        print("MuJoCo is required for the 6-DOF robot viewer and IK.")
        return

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data  = mujoco.MjData(model)
    hide_trajectory_markers(model)

    Q = None
    flags = None
    time_values = None
    angles = None
    camera_poses = None
    tcp_poses = None
    positions = None
    marker_positions = None
    playback_Q = None
    playback_time_values = None
    traj_info = None
    weld_poses = None

    # ── IK computation ───────────────────────────────────────────────────
    if not (args.ik or args.weld or args.camera or args.camera_viewer or args.verify or args.export_csv or args.no_viewer or args.joint_log or args.joint_log_full or args.collision_log):
        args.ik = True

    rotations = None
    if args.weld:
        Q, flags, time_values, angles, positions, weld_poses, _seam = compute_ik_weld_trajectory(
            model, data,
            verbose=not args.quiet_ik,
            retries=max(args.retries, WELD_DEFAULT_RETRIES),
            rng_seed=args.rng_seed,
        )
        camera_poses = weld_poses
        tcp_poses = weld_poses
        traj_info = None
    elif args.ik or args.camera or args.camera_viewer or args.verify or args.export_csv or args.joint_log or args.joint_log_full or args.collision_log:
        Q, flags, time_values, angles, positions, rotations, camera_poses, tcp_poses, traj_info = compute_ik_trajectory(
            model,
            data,
            verbose=not args.quiet_ik,
            retries=args.retries,
            rng_seed=args.rng_seed,
            max_waypoints=args.ik_waypoints,
        )

    if Q is not None and not args.no_viewer:
        if args.weld:
            # For welding, skip inspection-circle Cartesian playback and use
            # joint-space densification directly.
            playback_Q = _densify_joint_path(Q)
            playback_time_values = _retime_playback_path(playback_Q)
            print(
                f"[VIEWER-WELD] 관절공간 보간 플레이백: "
                f"{len(Q)} IK → {len(playback_Q)} samples"
            )
        else:
            playback_Q, playback_time_values, _ = build_capture_playback_trajectory(
                model,
                data,
                Q,
                time_values,
                traj_info=traj_info,
                playback_waypoints=args.playback_waypoints,
                retries=min(args.retries, PLAYBACK_CARTESIAN_RETRIES),
                rng_seed=args.rng_seed + 101,
            )

    if Q is not None:
        if args.joint_log or args.joint_log_full:
            print_joint_log(
                Q,
                flags,
                angles,
                traj_info=traj_info,
                full=args.joint_log_full,
                label="raw capture targets",
            )
            if playback_Q is not None and len(playback_Q) != len(Q):
                print_joint_log(
                    playback_Q,
                    np.ones(len(playback_Q), dtype=bool),
                    None,
                    traj_info=None,
                    full=False,
                    label="viewer playback continuous resampling",
                )
        if args.collision_log:
            print_collision_log(angles, traj_info=traj_info)
        if args.verify:
            verify_ik.model = model
            verify_ik.data = data
            verify_ik(Q, tcp_poses)
        if args.export_csv:
            save_trajectory_csv(args.export_csv, Q, flags, angles, camera_poses, tcp_poses, traj_info=traj_info)

    if args.no_viewer and not args.camera and not args.save_video:
        return

    # Place trajectory markers
    if positions is not None:
        marker_positions = positions
        update_trajectory_markers(model, data, marker_positions)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # ── Camera render ─────────────────────────────────────────────────────
    if args.camera and Q is not None:
        _cam_ok = render_inspection_cameras(model, data, Q, traj_info=traj_info, camera_name=args.camera_name)
        if not args.camera_viewer:
            if _cam_ok:
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
        Q=playback_Q if playback_Q is not None else Q,
        positions=marker_positions,
        time_values=playback_time_values if playback_time_values is not None else time_values,
        approach_duration=args.approach_duration,
        fixed_camera_name=args.camera_name if args.camera_viewer else None,
        show_frames=not args.hide_frames,
        overlay_poses=weld_poses if args.weld else None,
    )


if __name__ == "__main__":
    main()
