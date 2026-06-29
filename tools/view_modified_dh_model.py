#!/usr/bin/env python3
"""Open the Modified-DH generated robot model in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import mujoco.viewer

from model.franka import DH_PARAMS, TOOL_A, TOOL_ALPHA, TOOL_D, TOOL_THETA_OFFSET


ROBOT_READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571])
DEFAULT_MODEL_PATH = PROJECT_ROOT / "robot_model_modified_dh.xml"


def modified_dh_transform(d: float, theta: float, a_prev: float, alpha_prev: float) -> np.ndarray:
    """Modified DH transform Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
    ca, sa = np.cos(alpha_prev), np.sin(alpha_prev)
    ct, st = np.cos(theta), np.sin(theta)
    rx = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, ca, -sa, 0.0],
            [0.0, sa, ca, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    tx = np.eye(4)
    tx[0, 3] = a_prev
    rz = np.array(
        [
            [ct, -st, 0.0, 0.0],
            [st, ct, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    tz = np.eye(4)
    tz[2, 3] = d
    return rx @ tx @ rz @ tz


def fk_modified_dh(q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for qi, (d, theta_offset, a_prev, alpha_prev) in zip(q, DH_PARAMS):
        T = T @ modified_dh_transform(d, theta_offset + qi, a_prev, alpha_prev)
    return T @ modified_dh_transform(TOOL_D, TOOL_THETA_OFFSET, TOOL_A, TOOL_ALPHA)


def parse_q(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("q must contain 6 comma-separated values.")
    return np.asarray(values, dtype=float)


def joint_qpos_indices(model: mujoco.MjModel) -> list[int]:
    qidx: list[int] = []
    for i in range(1, 7):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        if joint_id < 0:
            raise RuntimeError(f"Missing joint{i} in model.")
        qidx.append(int(model.jnt_qposadr[joint_id]))
    return qidx


def set_q(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    qidx = joint_qpos_indices(model)
    data.qpos[qidx] = q
    for i in range(1, 7):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"q{i}_pos")
        if actuator_id >= 0:
            data.ctrl[actuator_id] = q[i - 1]
    mujoco.mj_forward(model, data)


def tcp_pose(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    if site_id < 0:
        raise RuntimeError("Missing tcp site in model.")
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[site_id]
    T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    return T


def print_pose_check(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> None:
    T_xml = tcp_pose(model, data)
    T_fk = fk_modified_dh(q)
    print("q =", np.array2string(q, precision=6, suppress_small=True))
    print("MuJoCo tcp position [m] =", np.array2string(T_xml[:3, 3], precision=6, suppress_small=True))
    print("MDH FK tcp position [m] =", np.array2string(T_fk[:3, 3], precision=6, suppress_small=True))
    print(f"position error = {np.linalg.norm(T_xml[:3, 3] - T_fk[:3, 3]):.3e} m")
    print(f"rotation error Frobenius = {np.linalg.norm(T_xml[:3, :3] - T_fk[:3, :3]):.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to Modified-DH MJCF XML.")
    parser.add_argument("--home", action="store_true", help="Show q=[0,0,0,0,0,0].")
    parser.add_argument("--ready", action="store_true", help="Show the ready pose used in the inspection scripts.")
    parser.add_argument("--q", type=parse_q, help="Custom joint values, e.g. --q '0,-0.8,0,-2.3,0,1.57'")
    parser.add_argument("--no-viewer", action="store_true", help="Only load the model and print TCP pose.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Generate robot_model_modified_dh.xml first."
        )

    if args.q is not None:
        q = args.q
    elif args.ready:
        q = ROBOT_READY.copy()
    else:
        q = np.zeros(6)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    set_q(model, data, q)
    print_pose_check(model, data, q)

    if args.no_viewer:
        return

    print("\n[VIEWER] Opening Modified-DH robot model.")
    print("  Axis colors: X=red, Y=green, Z=blue.")
    print("  Press [Esc] to exit.\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.sitegroup[:] = 1
        viewer.cam.lookat[:] = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")]
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    main()
