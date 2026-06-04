#!/usr/bin/env python3
"""
Run Franka Body-Jacobian CLIK on a segmented circular trajectory.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from control.clik import clik_step, solve_ik
from control.jacobian import body_jacobian
from model.franka import B_LIST, HOME_Q, M, Q_MAX, Q_MIN
from robot.kinematics import body_poe_fk
from trajectory.circle import segmented_circle_trajectory
from control.franka_ik_solver import set_arm_qpos, site_pose, solve_trajectory

try:
    import mujoco
    import mujoco.viewer

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None
    MUJOCO_AVAILABLE = False


SCENE_XML = os.path.join(os.path.dirname(__file__), "scene.xml")
MUJOCO_JOINT_NAMES = [f"panda0_joint{i}" for i in range(1, 8)]
FLANGE_CENTER = np.array([0.55, 0.0, 0.57])
TRAJECTORY_CENTER = np.array([0.37, 0.0, 0.57])
PIPE_OD = 0.0605
SEAM_RADIUS = PIPE_OD / 2.0
FRANKA_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0])
CAMERA_OFFSET_TCP_Z = 0.10
APPROACH_DURATION = 9.0
EE_LOOK_AXIS_SIGN = 1.0  # +z axis points toward the circle inside.
DEFAULT_PLAYBACK_SPEED = 1.0


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
    q_hist = np.zeros((len(time), 7))
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
    center: tuple[float, float, float] = (0.37, 0.0, 0.57),
    radius: float = 0.1652173913,
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
        "q": np.zeros((len(camera_positions), 7)),
        "desired_pos": camera_positions,
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
    q_min = np.full(7, -np.inf)
    q_max = np.full(7, np.inf)
    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo joint not found: {name}")
        if model.jnt_limited[joint_id]:
            q_min[i], q_max[i] = model.jnt_range[joint_id]
    return q_min, q_max


def _set_mujoco_arm_qpos(model, data, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=float).reshape(7)
    q_min, q_max = _mujoco_joint_ranges(model)
    q = np.clip(q, q_min, q_max)

    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = q[i]

        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = q[i]

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


def _prepend_zero_to_start(
    model,
    time_hist: np.ndarray,
    q_hist: np.ndarray,
    approach_duration: float = APPROACH_DURATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepend a smooth joint-space move from zero pose to the first waypoint."""
    q_min, q_max = _mujoco_joint_ranges(model)
    q_zero = np.clip(np.zeros(7), q_min, q_max)
    q_start = q_hist[0]

    if approach_duration <= 0.0:
        return time_hist, q_hist

    nominal_dt = float(np.median(np.diff(time_hist))) if len(time_hist) > 1 else 0.02
    nominal_dt = max(nominal_dt, 0.01)
    n_steps = max(2, int(np.ceil(approach_duration / nominal_dt)))
    approach_time = np.linspace(0.0, approach_duration, n_steps + 1)
    u = approach_time / approach_duration
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    approach_q = q_zero[None, :] + s[:, None] * (q_start - q_zero)[None, :]

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
        T_tcp[:3, 3] = camera_pos + EE_LOOK_AXIS_SIGN * CAMERA_OFFSET_TCP_Z * look_dir
        tcp_poses.append(T_tcp)
    return tcp_poses


def _camera_poses_to_tcp_poses(camera_poses: np.ndarray) -> list[np.ndarray]:
    tcp_poses = []
    for T_cam in np.asarray(camera_poses, dtype=float):
        T_tcp = T_cam.copy()
        T_tcp[:3, 3] = T_cam[:3, 3] + EE_LOOK_AXIS_SIGN * CAMERA_OFFSET_TCP_Z * T_cam[:3, 2]
        T_tcp[:3, 2] = EE_LOOK_AXIS_SIGN * T_cam[:3, 2]
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


def run_mujoco_viewer(
    log: dict,
    scene_path: str = SCENE_XML,
    playback_speed: float = DEFAULT_PLAYBACK_SPEED,
    retries: int = 16,
    max_waypoints: int = 240,
    approach_duration: float = APPROACH_DURATION,
) -> None:
    if not MUJOCO_AVAILABLE:
        print("MuJoCo가 설치되어 있지 않습니다. 설치: pip install mujoco")
        return
    if not os.path.exists(scene_path):
        raise FileNotFoundError(scene_path)

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    time_all = np.asarray(log["time"], dtype=float)
    desired_all = np.asarray(log["desired_pos"], dtype=float)
    look_targets_all = np.asarray(
        log.get("look_targets", np.tile(FLANGE_CENTER.reshape(1, 3), (len(desired_all), 1))),
        dtype=float,
    )
    if len(desired_all) > max_waypoints:
        sample_idx = np.linspace(0, len(desired_all) - 1, max_waypoints).round().astype(int)
        time_hist = time_all[sample_idx]
        desired_camera_pos = desired_all[sample_idx]
        look_targets = look_targets_all[sample_idx]
    else:
        time_hist = time_all
        desired_camera_pos = desired_all
        look_targets = look_targets_all

    _update_mujoco_reference_markers(model, desired_all)

    print("\n[MuJoCo IK] 파란 reference 점에 camera_optical_center를 맞추는 IK를 계산합니다.")
    tcp_poses = _camera_positions_to_tcp_poses(desired_camera_pos, look_targets)
    q_hist, flags = solve_trajectory(
        model,
        data,
        mujoco,
        tcp_poses,
        look_target=look_targets,
        q_start=FRANKA_READY,
        retries=retries,
        verbose=False,
        axis_col=2,
        axis_sign=EE_LOOK_AXIS_SIGN,
    )

    actual_camera_pos = np.zeros_like(desired_camera_pos)
    for i, q in enumerate(q_hist):
        set_arm_qpos(model, data, mujoco, q)
        actual_camera_pos[i] = site_pose(model, data, mujoco, "camera_optical_center")[:3, 3]

    camera_err = np.linalg.norm(actual_camera_pos - desired_camera_pos, axis=1)
    print(f"[MuJoCo IK] success: {sum(flags)}/{len(flags)}")
    print(f"[MuJoCo IK] max camera position error: {np.max(camera_err) * 1000:.2f} mm")

    time_hist, q_hist = _prepend_zero_to_start(
        model,
        time_hist,
        q_hist,
        approach_duration=approach_duration,
    )
    _set_mujoco_arm_qpos(model, data, q_hist[0])

    playback_speed = max(float(playback_speed), 1e-6)

    print("\n[MuJoCo] viewer를 실행합니다.")
    print("[MuJoCo] scene.xml Franka 기준 IK trajectory를 재생합니다.")
    print("[MuJoCo] 창을 닫으면 프로그램이 종료됩니다.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = FLANGE_CENTER
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 45

        wall_t0 = time.time()
        while viewer.is_running():
            elapsed = (time.time() - wall_t0) * playback_speed
            traj_t = min(time_hist[0] + elapsed, time_hist[-1])
            q = _interpolate_q(time_hist, q_hist, traj_t)

            _set_mujoco_arm_qpos(model, data, q)
            viewer.sync()
            time.sleep(0.002)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", nargs=3, type=float, default=(0.37, 0.0, 0.57))
    parser.add_argument("--radius", type=float, default=0.1652173913)
    parser.add_argument("--segment-duration", type=float, default=9.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--save", default=None, help="Optional path for the matplotlib figure.")
    parser.add_argument("--no-show", action="store_true", help="Run without opening a plot window.")
    parser.add_argument("--mujoco", action="store_true", help="Replay the CLIK trajectory in MuJoCo viewer.")
    parser.add_argument("--scene", default=SCENE_XML, help="MuJoCo XML scene path.")
    parser.add_argument("--playback-speed", type=float, default=DEFAULT_PLAYBACK_SPEED, help="MuJoCo replay speed multiplier.")
    parser.add_argument("--retries", type=int, default=16, help="MuJoCo IK random restarts per waypoint.")
    parser.add_argument("--mujoco-waypoints", type=int, default=240, help="Number of MuJoCo IK waypoints sampled from the reference.")
    parser.add_argument("--approach-duration", type=float, default=APPROACH_DURATION, help="Seconds to move from zero pose to the 12 o'clock start.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mujoco:
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
        )
        return

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


if __name__ == "__main__":
    main()
