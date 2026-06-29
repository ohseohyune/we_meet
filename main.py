#!/usr/bin/env python3
"""
Run 6-DOF DH Body-Jacobian CLIK on a segmented circular trajectory.
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import numpy as np

from control.clik import clik_step, solve_ik
from control.jacobian import body_jacobian
from model.franka import B_LIST, HOME_Q, M, Q_MAX, Q_MIN
from robot.kinematics import body_poe_fk
from trajectory.circle import segmented_circle_trajectory
from control.franka_ik_solver import (
    retime_joint_trajectory,
    set_arm_qpos,
    site_pose,
    solve_trajectory,
)
from viz import (
    Dashboard,
    Renderer3D,
    RingLogger,
    Skeleton3D,
    dashboard_available,
    save_summary_png,
    skeleton3d_available,
)

try:
    import mujoco
    import mujoco.viewer

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None
    MUJOCO_AVAILABLE = False


SCENE_XML = os.path.join(os.path.dirname(__file__), "scene.xml")
MUJOCO_JOINT_NAMES = [f"q{i}" for i in range(1, 7)]
MUJOCO_ACTUATOR_NAMES = [f"p{i}" for i in range(1, 7)]
FLANGE_CENTER = np.array([0.67000, 0.0, 0.5286])
TRAJECTORY_CENTER = np.array([0.51670, 0.0, 0.5286])
PIPE_OD = 0.0605
SEAM_RADIUS = PIPE_OD / 2.0
ROBOT_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571])
RIGHT_BIASED_READY = np.array([1.50, -0.785, 0.40, -2.356, 0.0, 1.571])
CAMERA_SITE_NAME = "camera_optical_center"
CAMERA_OFFSET_FROM_TIP_EE = np.array([-0.00841, -0.03657, 0.08456])
APPROACH_DURATION = 9.0
MAX_JOINT_SPEED = 0.85
EE_LOOK_AXIS_COL = 2
EE_LOOK_AXIS_SIGN = -1.0  # MuJoCo camera optical axis is camera-site -z.
CAMERA_IN_EE_ROT = np.array(
    [
        [0.99463114, 0.01241430, -0.10273648],
        [-0.04061714, 0.95994352, -0.27723400],
        [0.09517956, 0.27991844, 0.95529395],
    ]
)
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_OUT_DIR = "out"
DEFAULT_DASHBOARD_WINDOW = 10.0
DEFAULT_RECORD_FPS = 30.0
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
TRAJECTORY_MARKER_RGBA = np.array([0.0, 0.85, 1.0, 1.0])
TRAJECTORY_MARKER_SIZE = 0.014


def run_tracking(
    center: tuple[float, float, float],
    radius: float,
    segment_duration: float,
    dt: float,
) -> dict:
    traj = segmented_circle_trajectory(
        center=center,
        radius=radius,
        segment_duration=segment_duration,
        dt=dt,
        orientation_target=FLANGE_CENTER,
        target_radius=SEAM_RADIUS,
    )
    poses = traj["poses"]
    time = traj["time"]

    K_init = np.diag([4.0, 4.0, 4.0, 7.0, 7.0, 7.0])
    q, success, ik_info = solve_ik(
        poses[0],
        B_LIST,
        M,
        HOME_Q,
        K_p=K_init,
        max_iter=500,
        tol=2e-4,
        dt=0.035,
        damping=0.06,
        q_lo=Q_MIN,
        q_hi=Q_MAX,
    )

    K_track = np.diag([6.0, 6.0, 6.0, 10.0, 10.0, 10.0])
    q_hist = np.zeros((len(time), len(HOME_Q)))
    actual_pos = np.zeros((len(time), 3))
    desired_pos = traj["positions"].copy()
    error_norm = np.zeros(len(time))
    cond_hist = np.zeros(len(time))

    for k, T_des in enumerate(poses):
        q, info = clik_step(
            q,
            T_des,
            B_LIST,
            M,
            K_track,
            dt,
            damping=0.05,
            q_lo=Q_MIN,
            q_hi=Q_MAX,
            nullspace_gain=0.06,
            return_info=True,
        )
        T_cur = body_poe_fk(q, B_LIST, M)

        q_hist[k] = q
        actual_pos[k] = T_cur[:3, 3]
        error_norm[k] = info["error_norm"]
        cond_hist[k] = np.linalg.cond(body_jacobian(q, B_LIST))

    return {
        "time": time,
        "q": q_hist,
        "desired_pos": desired_pos,
        "actual_pos": actual_pos,
        "error_norm": error_norm,
        "condition": cond_hist,
        "initial_ik_success": success,
        "initial_ik_info": ik_info,
        "trajectory": traj,
    }


def run_viewer_reference(
    center: tuple[float, float, float] = (0.51670, 0.0, 0.5286),
    radius: float = 0.1200,
    segment_duration: float = 9.0,
    dt: float = 0.02,
) -> dict:
    """Segmented top-bottom-top-bottom-top reference for the MuJoCo viewer."""
    traj = segmented_circle_trajectory(
        center=center,
        radius=radius,
        segment_duration=segment_duration,
        dt=dt,
        orientation_target=FLANGE_CENTER,
        target_radius=SEAM_RADIUS,
    )
    camera_positions = traj["positions"]
    return {
        "time": traj["time"],
        "q": np.zeros((len(camera_positions), len(HOME_Q))),
        "desired_pos": camera_positions,
        "desired_pose": traj["poses"],
        "look_targets": traj["targets"],
        "actual_pos": np.full_like(camera_positions, np.nan),
        "error_norm": np.full(len(camera_positions), np.nan),
        "condition": np.full(len(camera_positions), np.nan),
        "initial_ik_success": True,
        "initial_ik_info": {},
        "trajectory": traj,
    }


def plot_results(log: dict, save_path: str | None = None, show: bool = True) -> None:
    time = log["time"]
    q = log["q"]
    desired = log["desired_pos"]
    actual = log["actual_pos"]

    fig = plt.figure(figsize=(13, 9))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], "k--", label="desired")
    ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], "tab:blue", label="actual")
    ax.scatter(desired[0, 0], desired[0, 1], desired[0, 2], c="green", s=40, label="start")
    ax.scatter(desired[-1, 0], desired[-1, 1], desired[-1, 2], c="red", s=40, label="end")
    ax.set_title("End-Effector Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(2, 2, 2)
    for j in range(q.shape[1]):
        ax.plot(time, q[:, j], label=f"q{j + 1}")
    ax.set_title("Joint Angle History")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint angle [rad]")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)

    ax = fig.add_subplot(2, 2, 3)
    ax.semilogy(time, log["error_norm"])
    ax.set_title("Body Error Norm")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||e_b||")
    ax.grid(True)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(time, log["condition"])
    ax.set_title("Body Jacobian Condition Number")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("cond(J_b)")
    ax.grid(True)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _mujoco_joint_ranges(model) -> tuple[np.ndarray, np.ndarray]:
    q_min = np.full(len(MUJOCO_JOINT_NAMES), -np.inf)
    q_max = np.full(len(MUJOCO_JOINT_NAMES), np.inf)
    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo joint not found: {name}")
        if model.jnt_limited[joint_id]:
            q_min[i], q_max[i] = model.jnt_range[joint_id]
    return q_min, q_max


def _set_mujoco_arm_qpos(model, data, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=float).reshape(len(MUJOCO_JOINT_NAMES))
    q_min, q_max = _mujoco_joint_ranges(model)

    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = q[i]

        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, MUJOCO_ACTUATOR_NAMES[i])
        if actuator_id >= 0:
            data.ctrl[actuator_id] = np.clip(q[i], q_min[i], q_max[i])

    mujoco.mj_forward(model, data)


def _interpolate_q(time_hist: np.ndarray, q_hist: np.ndarray, t: float) -> np.ndarray:
    if t <= time_hist[0]:
        return q_hist[0]
    if t >= time_hist[-1]:
        return q_hist[-1]

    hi = int(np.searchsorted(time_hist, t, side="right"))
    lo = hi - 1
    span = max(time_hist[hi] - time_hist[lo], 1e-12)
    alpha = (t - time_hist[lo]) / span
    return (1.0 - alpha) * q_hist[lo] + alpha * q_hist[hi]


def _reference_position_at(time_hist: np.ndarray, positions: np.ndarray, t: float) -> np.ndarray:
    if len(time_hist) == 0 or len(positions) == 0:
        return np.full(3, np.nan)
    t_clip = float(np.clip(t, time_hist[0], time_hist[-1]))
    return _interpolate_q(time_hist, positions, t_clip)


def _prepend_zero_to_start(
    model,
    time_hist: np.ndarray,
    q_hist: np.ndarray,
    approach_duration: float = APPROACH_DURATION,
    via_q: np.ndarray | None = RIGHT_BIASED_READY,
    via_fraction: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend a smooth joint-space move through a right-side posture."""
    q_min, q_max = _mujoco_joint_ranges(model)
    q_zero = np.clip(RIGHT_BIASED_READY.copy(), q_min, q_max)
    q_start = q_hist[0]

    if approach_duration <= 0.0:
        return time_hist, q_hist

    nominal_dt = float(np.median(np.diff(time_hist))) if len(time_hist) > 1 else 0.02
    nominal_dt = max(nominal_dt, 0.01)
    n_steps = max(2, int(np.ceil(approach_duration / nominal_dt)))

    if via_q is None:
        approach_time = np.linspace(0.0, approach_duration, n_steps + 1)
        u = approach_time / approach_duration
        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        approach_q = q_zero[None, :] + s[:, None] * (q_start - q_zero)[None, :]
    else:
        q_via = np.clip(np.asarray(via_q, dtype=float), q_min, q_max)
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

    shifted_time = approach_duration + (time_hist - time_hist[0])
    return (
        np.concatenate([approach_time, shifted_time[1:]]),
        np.vstack([approach_q, q_hist[1:]]),
    )


def _camera_positions_to_tcp_poses(
    camera_positions: np.ndarray,
    look_targets: np.ndarray | None = None,
) -> list[np.ndarray]:
    camera_positions = np.asarray(camera_positions, dtype=float)
    if look_targets is None:
        look_targets = np.tile(FLANGE_CENTER.reshape(1, 3), (len(camera_positions), 1))
    look_targets = np.asarray(look_targets, dtype=float)

    tcp_poses = []
    for camera_pos, look_target in zip(camera_positions, look_targets):
        look_dir = look_target - camera_pos
        look_dir /= np.linalg.norm(look_dir) + 1e-12

        T_tcp = np.eye(4)
        T_tcp[:3, 3] = camera_pos
        tcp_poses.append(T_tcp)
    return tcp_poses


def _camera_poses_to_tcp_poses(camera_poses: np.ndarray) -> list[np.ndarray]:
    tcp_poses = []
    for T_cam in np.asarray(camera_poses, dtype=float):
        T_tcp = T_cam.copy()
        T_tcp[:3, 3] = T_cam[:3, 3]
        T_tcp[:3, :3] = T_cam[:3, :3]
        tcp_poses.append(T_tcp)
    return tcp_poses


def _update_mujoco_reference_markers(model, positions: np.ndarray) -> None:
    """Map the dense reference trajectory to the wp_00... markers in scene.xml."""
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
        model.site_rgba[site_id] = TRAJECTORY_MARKER_RGBA
        model.site_size[site_id] = TRAJECTORY_MARKER_SIZE


def _right_posture_weights(log: dict, sample_idx: np.ndarray | None, n_waypoints: int) -> np.ndarray:
    traj = log.get("trajectory", {})
    segment_ids = np.asarray(traj.get("segment_id", []), dtype=int)
    desired_len = len(np.asarray(log.get("desired_pos", [])))
    if segment_ids.shape != (desired_len,):
        return np.zeros(n_waypoints, dtype=float)
    if sample_idx is not None:
        segment_ids = segment_ids[sample_idx]
    if segment_ids.shape != (n_waypoints,):
        return np.zeros(n_waypoints, dtype=float)
    return np.array(
        [RIGHT_POSTURE_SEGMENT_WEIGHTS.get(int(segment_id), 0.0) for segment_id in segment_ids],
        dtype=float,
    )


def run_mujoco_viewer(
    log: dict,
    scene_path: str = SCENE_XML,
    playback_speed: float = DEFAULT_PLAYBACK_SPEED,
    retries: int = 16,
    max_waypoints: int = 240,
    approach_duration: float = APPROACH_DURATION,
    mode: str = "live",
    out_dir: str = DEFAULT_OUT_DIR,
    dashboard_window: float = DEFAULT_DASHBOARD_WINDOW,
    heatmap: bool = False,
    fixed_y: bool = False,
    skeleton: bool = True,
    record_fps: float = DEFAULT_RECORD_FPS,
    render_width: int = 1280,
    render_height: int = 720,
) -> None:
    if not MUJOCO_AVAILABLE:
        print("MuJoCo가 설치되어 있지 않습니다. 설치: pip install mujoco")
        return
    if not os.path.exists(scene_path):
        raise FileNotFoundError(scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # --- 모델 검증 ---
    if model.nq != 6 or model.nu != 6:
        raise RuntimeError(f"모델 nq={model.nq}, nu={model.nu} — 6-DOF 모델이 아닙니다.")
    for jname in MUJOCO_JOINT_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname) < 0:
            raise RuntimeError(f"조인트 '{jname}'를 모델에서 찾을 수 없습니다.")
    for aname in MUJOCO_ACTUATOR_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname) < 0:
            raise RuntimeError(f"액추에이터 '{aname}'를 모델에서 찾을 수 없습니다.")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee") < 0:
        raise RuntimeError("body 'ee'를 모델에서 찾을 수 없습니다.")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp") < 0:
        raise RuntimeError("site 'tcp'를 모델에서 찾을 수 없습니다.")

    mujoco.mj_forward(model, data)
    tcp_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")].copy()
    if np.any(np.isnan(tcp_pos)):
        raise RuntimeError("mj_forward 후 TCP 위치에 NaN이 검출되었습니다.")
    print(f"[검증] nq=6, nu=6, ee body, tcp site 확인 완료")
    print(f"[검증] TCP 초기 위치 (q=0): {tcp_pos}")

    ee_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_cam")
    if ee_cam_id < 0:
        raise RuntimeError("camera 'ee_cam'를 모델에서 찾을 수 없습니다.")
    cam_dist = np.linalg.norm(
        data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "camera_optical_center")]
        - data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee")]
    )
    print(f"[검증] ee_cam id={ee_cam_id}, EE에서 카메라까지 거리={cam_dist*100:.1f} cm")
    time_all = np.asarray(log["time"], dtype=float)
    desired_all = np.asarray(log["desired_pos"], dtype=float)
    desired_poses_all = np.asarray(log.get("desired_pose", []), dtype=float)
    look_targets_all = np.asarray(
        log.get("look_targets", np.tile(FLANGE_CENTER.reshape(1, 3), (len(desired_all), 1))),
        dtype=float,
    )
    sample_idx = None
    if len(desired_all) > max_waypoints:
        sample_idx = np.linspace(0, len(desired_all) - 1, max_waypoints).round().astype(int)
        time_hist = time_all[sample_idx]
        desired_camera_pos = desired_all[sample_idx]
        desired_camera_poses = desired_poses_all[sample_idx] if len(desired_poses_all) else None
        look_targets = look_targets_all[sample_idx]
    else:
        time_hist = time_all
        desired_camera_pos = desired_all
        desired_camera_poses = desired_poses_all if len(desired_poses_all) else None
        look_targets = look_targets_all

    _update_mujoco_reference_markers(model, desired_all)
    posture_weights = _right_posture_weights(log, sample_idx, len(desired_camera_pos))

    print("\n[MuJoCo IK] 파란 reference 점에 camera_optical_center를 맞추는 IK를 계산합니다.")
    if desired_camera_poses is not None:
        tcp_poses = _camera_poses_to_tcp_poses(desired_camera_poses)
    else:
        tcp_poses = _camera_positions_to_tcp_poses(desired_camera_pos, look_targets)
    q_hist, flags = solve_trajectory(
        model,
        data,
        mujoco,
        tcp_poses,
        look_target=look_targets,
        q_start=RIGHT_BIASED_READY,
        retries=retries,
        verbose=False,
        axis_col=EE_LOOK_AXIS_COL,
        axis_sign=EE_LOOK_AXIS_SIGN,
        posture_bias=RIGHT_POSTURE_BIAS,
        posture_weights=posture_weights,
        site_name=CAMERA_SITE_NAME,
    )

    actual_camera_pos = np.zeros_like(desired_camera_pos)
    for i, q in enumerate(q_hist):
        set_arm_qpos(model, data, mujoco, q)
        actual_camera_pos[i] = site_pose(model, data, mujoco, CAMERA_SITE_NAME)[:3, 3]

    camera_err = np.linalg.norm(actual_camera_pos - desired_camera_pos, axis=1)
    print(f"[MuJoCo IK] success: {sum(flags)}/{len(flags)}")
    print(f"[MuJoCo IK] max camera position error: {np.max(camera_err) * 1000:.2f} mm")

    original_duration = float(time_hist[-1] - time_hist[0]) if len(time_hist) > 1 else 0.0
    time_hist = retime_joint_trajectory(time_hist, q_hist, max_joint_speed=MAX_JOINT_SPEED)
    retimed_duration = float(time_hist[-1] - time_hist[0]) if len(time_hist) > 1 else 0.0
    if retimed_duration > original_duration + 1e-6:
        print(
            f"[MuJoCo IK] Retimed for max joint speed {MAX_JOINT_SPEED:.2f} rad/s: "
            f"{original_duration:.2f}s -> {retimed_duration:.2f}s"
        )

    time_hist, q_hist = _prepend_zero_to_start(
        model,
        time_hist,
        q_hist,
        approach_duration=approach_duration,
    )
    _set_mujoco_arm_qpos(model, data, q_hist[0])

    playback_speed = max(float(playback_speed), 1e-6)

    os.makedirs(out_dir, exist_ok=True)
    log_csv = os.path.join(out_dir, "log.csv")
    summary_png = os.path.join(out_dir, "summary.png")
    sim_mp4 = os.path.join(out_dir, "sim.mp4")
    logger = RingLogger(window_seconds=dashboard_window, nominal_dt=float(model.opt.timestep))
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    skeleton_body_names = ("link1", "link2", "link3", "link4", "link5", "link6", "ee")
    skeleton_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for body_name in skeleton_body_names
    ]
    skeleton_body_names = tuple(
        body_name for body_name, body_id in zip(skeleton_body_names, skeleton_body_ids) if body_id >= 0
    )
    skeleton_body_ids = [body_id for body_id in skeleton_body_ids if body_id >= 0]
    ref_data = mujoco.MjData(model)

    def tcp_ref_from_q(q_ref: np.ndarray) -> np.ndarray:
        ref_data.qpos[:6] = np.asarray(q_ref, dtype=float).reshape(6)
        mujoco.mj_forward(model, ref_data)
        return ref_data.site_xpos[tcp_site_id].copy()

    ref_tcp_path = np.vstack([tcp_ref_from_q(q_ref) for q_ref in q_hist])

    requested_mode = mode
    dashboard_ok = dashboard_available()
    skeleton_ok = skeleton and skeleton3d_available()
    if mode == "live" and not dashboard_ok:
        print("[경고] pyqtgraph/Qt를 사용할 수 없어 dashboard만 비활성화합니다. MuJoCo viewer는 계속 실행합니다.")
    if mode == "live" and skeleton and not skeleton_ok:
        print("[경고] pyqtgraph/OpenGL을 사용할 수 없어 별도 3D skeleton 창을 비활성화합니다.")

    renderer = Renderer3D(
        model,
        data,
        mujoco,
        mode=mode,
        ref_path=ref_tcp_path,
        out_video=sim_mp4,
        width=render_width,
        height=render_height,
        fps=record_fps,
        fixed_camera_id=None if mode == "live" else ee_cam_id,
        ghost=False,
        free_camera_lookat=FLANGE_CENTER,
        overlay_enabled=(mode == "record"),
    )
    dashboard = None
    skeleton_view = None
    try:
        renderer.start()
    except Exception as exc:
        if requested_mode == "live":
            print(f"[경고] passive viewer 초기화 실패({exc}). --record 모드로 자동 전환합니다.")
            mode = "record"
            renderer.close()
            renderer = Renderer3D(
                model,
                data,
                mujoco,
                mode="record",
                ref_path=ref_tcp_path,
                out_video=sim_mp4,
                width=render_width,
                height=render_height,
                fps=record_fps,
                fixed_camera_id=ee_cam_id,
                ghost=False,
                free_camera_lookat=FLANGE_CENTER,
                overlay_enabled=True,
            )
            renderer.start()
        else:
            raise

    if mode == "live" and dashboard_ok:
        dashboard = Dashboard(window_seconds=dashboard_window, heatmap=heatmap, fixed_y=fixed_y)
        if not dashboard.start():
            print("[경고] pyqtgraph dashboard 시작 실패. dashboard만 비활성화하고 MuJoCo viewer는 계속 실행합니다.")
            dashboard = None

    if mode == "live" and skeleton_ok:
        skeleton_view = Skeleton3D(ref_tcp_path, body_names=skeleton_body_names)
        if not skeleton_view.start():
            print("[경고] 3D skeleton 창 시작 실패. MuJoCo viewer는 계속 실행합니다.")
            skeleton_view = None

    print("\n[MuJoCo] generated_robot.xml 6-DOF DH robot IK trajectory를 재생합니다.")
    print(f"[MuJoCo] mode={mode}, log={log_csv}")
    if mode == "record":
        print(f"[MuJoCo] offscreen video={sim_mp4}")
    else:
        if dashboard is None:
            print("[MuJoCo] passive viewer는 mujoco_viewer.py --ik와 같은 free-camera 화면으로 유지합니다.")
        else:
            print("[MuJoCo] passive viewer + pyqtgraph dashboard. MuJoCo 창은 원본 viewer 화면으로 유지합니다.")
        if skeleton_view is None:
            print("[MuJoCo] 별도 3D skeleton 창은 비활성화되었습니다.")
        else:
            print("[MuJoCo] 별도 3D skeleton + reference trajectory 창을 실행했습니다.")
    print("[가정] q_ref는 기존 재생 루프의 q 변수이며, 명시적 q_ref가 없으면 data.ctrl[:6]가 같은 값으로 설정됩니다.\n")

    steps = 0
    wall_t0 = time.time()
    try:
        if mode == "live":
            while renderer.is_running():
                elapsed = (time.time() - wall_t0) * playback_speed
                traj_t = min(time_hist[0] + elapsed, time_hist[-1])
                q = _interpolate_q(time_hist, q_hist, traj_t)

                _set_mujoco_arm_qpos(model, data, q)
                p_ref = tcp_ref_from_q(q)
                p_tcp = data.site_xpos[tcp_site_id].copy()
                logger.log(traj_t, q, data.qpos[:6], p_tcp, p_ref=p_ref)
                progress = (traj_t - time_hist[0]) / max(time_hist[-1] - time_hist[0], 1.0e-9)
                renderer.sync(tcp_pos=p_tcp, target_pos=p_ref, q_ref=q, progress=progress)
                if dashboard is not None:
                    dashboard.push(logger.snapshot())
                if skeleton_view is not None:
                    skeleton_view.push(
                        {
                            "body_pos": data.xpos[skeleton_body_ids].copy(),
                            "tcp_pos": p_tcp,
                            "target_pos": p_ref,
                            "progress": progress,
                        }
                    )
                steps += 1
                if traj_t >= time_hist[-1]:
                    wall_t0 = time.time()
                time.sleep(float(model.opt.timestep))
        else:
            sim_dt = max(float(model.opt.timestep), 1.0e-4)
            t = float(time_hist[0])
            t_end = float(time_hist[-1])
            next_frame_t = t
            frame_dt = 1.0 / max(float(record_fps), 1.0)
            while t <= t_end + 0.5 * sim_dt:
                q = _interpolate_q(time_hist, q_hist, t)
                _set_mujoco_arm_qpos(model, data, q)
                p_ref = tcp_ref_from_q(q)
                p_tcp = data.site_xpos[tcp_site_id].copy()
                logger.log(t, q, data.qpos[:6], p_tcp, p_ref=p_ref)
                if t >= next_frame_t - 0.5 * sim_dt:
                    progress = (t - time_hist[0]) / max(time_hist[-1] - time_hist[0], 1.0e-9)
                    renderer.render_frame(tcp_pos=p_tcp, target_pos=p_ref, q_ref=q, progress=progress)
                    next_frame_t += frame_dt
                steps += 1
                t += sim_dt
    finally:
        if dashboard is not None:
            dashboard.close()
        if skeleton_view is not None:
            skeleton_view.close()
        renderer.close()
        logger.save_csv(log_csv)
        save_summary_png(logger.records(), summary_png)

    wall_elapsed = max(time.time() - wall_t0, 1.0e-9)
    achieved_steps = steps / wall_elapsed
    nominal_steps = 1.0 / max(float(model.opt.timestep), 1.0e-9)
    slowdown = 1.0 - (achieved_steps / nominal_steps)
    summary = logger.summary()
    print(f"[검증] samples={summary.samples}, duration={summary.duration:.3f}s")
    print("[검증] joint RMSE [rad]: " + " ".join(f"{v:.6f}" for v in summary.joint_rmse))
    print(f"[검증] ||e_q|| RMSE={summary.joint_norm_rmse:.6f} rad")
    if summary.final_tcp_error is not None:
        print(
            "[검증] workspace error ||e_x|| [m]: "
            f"final={summary.final_tcp_error:.6f}, mean={summary.mean_tcp_error:.6f}, max={summary.max_tcp_error:.6f}"
        )
    print(f"[검증] output: {sim_mp4 if mode == 'record' else '(live viewer)'}, {log_csv}, {summary_png}")
    print(f"[성능] achieved={achieved_steps:.1f} steps/s, nominal_unattached={nominal_steps:.1f} steps/s, slowdown={slowdown*100:.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", nargs=3, type=float, default=(0.51670, 0.0, 0.5286))
    parser.add_argument("--radius", type=float, default=0.1200)
    parser.add_argument("--segment-duration", type=float, default=9.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--save", default=None, help="Optional path for the matplotlib figure.")
    parser.add_argument("--no-show", action="store_true", help="Run without opening a plot window.")
    parser.add_argument("--analysis", action="store_true", help="Run the legacy CLIK analysis plot instead of visualization.")
    parser.add_argument("--live", action="store_true", help="Run passive viewer + pyqtgraph dashboard (default).")
    parser.add_argument("--record", action="store_true", help="Run headless offscreen rendering and save out/sim.mp4.")
    parser.add_argument("--mujoco", action="store_true", help="Alias for --live.")
    parser.add_argument("--scene", default=SCENE_XML, help="MuJoCo XML scene path.")
    parser.add_argument("--playback-speed", type=float, default=DEFAULT_PLAYBACK_SPEED, help="MuJoCo replay speed multiplier.")
    parser.add_argument("--retries", type=int, default=16, help="MuJoCo IK random restarts per waypoint.")
    parser.add_argument("--mujoco-waypoints", type=int, default=240, help="Number of MuJoCo IK waypoints sampled from the reference.")
    parser.add_argument("--approach-duration", type=float, default=APPROACH_DURATION, help="Seconds to move from zero pose to the 12 o'clock start.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for log.csv, sim.mp4, and summary.png.")
    parser.add_argument("--dashboard-window", type=float, default=DEFAULT_DASHBOARD_WINDOW, help="Dashboard sliding window in seconds.")
    parser.add_argument("--heatmap", action="store_true", help="Show joint x time error heatmap in the dashboard.")
    parser.add_argument("--fixed-y", action="store_true", help="Start the dashboard with fixed y-axis ranges.")
    parser.add_argument("--skeleton", dest="skeleton", action="store_true", help="Open the separate 3D skeleton + reference trajectory window.")
    parser.add_argument("--no-skeleton", dest="skeleton", action="store_false", help="Disable the separate 3D skeleton window.")
    parser.set_defaults(skeleton=True)
    parser.add_argument("--record-fps", type=float, default=DEFAULT_RECORD_FPS, help="Offscreen video frame rate.")
    parser.add_argument("--render-width", type=int, default=1280, help="Offscreen render width.")
    parser.add_argument("--render-height", type=int, default=720, help="Offscreen render height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.analysis:
        log = run_tracking(
            center=tuple(args.center),
            radius=args.radius,
            segment_duration=args.segment_duration,
            dt=args.dt,
        )

        print(f"initial IK success: {log['initial_ik_success']}")
        print(f"final body error norm: {log['error_norm'][-1]:.6f}")
        print(f"max body error norm: {np.max(log['error_norm']):.6f}")
        print(f"max cond(J_b): {np.max(log['condition']):.3f}")

        plot_results(log, save_path=args.save, show=not args.no_show)
        return

    if args.live or args.record or args.mujoco or not args.analysis:
        log = run_viewer_reference(
            center=tuple(args.center),
            radius=args.radius,
            segment_duration=args.segment_duration,
            dt=args.dt,
        )
        run_mujoco_viewer(
            log,
            scene_path=args.scene,
            playback_speed=args.playback_speed,
            retries=args.retries,
            max_waypoints=args.mujoco_waypoints,
            approach_duration=args.approach_duration,
            mode="record" if args.record else "live",
            out_dir=args.out_dir,
            dashboard_window=args.dashboard_window,
            heatmap=args.heatmap,
            fixed_y=args.fixed_y,
            skeleton=args.skeleton,
            record_fps=args.record_fps,
            render_width=args.render_width,
            render_height=args.render_height,
        )
        return


if __name__ == "__main__":
    main()
