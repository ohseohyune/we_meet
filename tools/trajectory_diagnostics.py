#!/usr/bin/env python3
"""Diagnose trajectory joint jumps and Jacobian singularity indicators."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control.franka_ik_solver import joint_limits, joint_qpos_indices, set_arm_qpos, solve_trajectory
from main import (
    EE_LOOK_AXIS_COL,
    EE_LOOK_AXIS_SIGN,
    ROBOT_READY,
    SCENE_XML,
)
from mujoco_viewer import CAMERA_SITE_NAME, camera_poses_to_tcp_poses, generate_segmented_reference


def tcp_jacobian(model, data, q: np.ndarray) -> np.ndarray:
    qidx = joint_qpos_indices(model, mujoco)
    set_arm_qpos(model, data, mujoco, q)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CAMERA_SITE_NAME)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)
    return np.vstack([jacp[:, qidx], jacr[:, qidx]])


def jacobian_metrics(J: np.ndarray) -> dict:
    s = np.linalg.svd(J, compute_uv=False)
    min_sv = float(np.min(s))
    max_sv = float(np.max(s))
    cond = float(max_sv / max(min_sv, 1e-12))
    gram = J @ J.T
    sign, logabsdet = np.linalg.slogdet(gram)
    det_jjt = float(sign * np.exp(logabsdet)) if sign > 0 else 0.0
    manipulability = float(np.sqrt(max(det_jjt, 0.0)))
    return {
        "min_sv": min_sv,
        "max_sv": max_sv,
        "condition": cond,
        "det_jjt": det_jjt,
        "logdet_jjt": float(logabsdet) if sign > 0 else -np.inf,
        "manipulability": manipulability,
    }


def limit_metrics(q: np.ndarray, limits: np.ndarray) -> dict:
    lower_margin = q - limits[:, 0]
    upper_margin = limits[:, 1] - q
    margins = np.minimum(lower_margin, upper_margin)
    closest = int(np.argmin(margins))
    return {
        "limit_margin_min": float(margins[closest]),
        "limit_closest_joint": closest + 1,
        "limit_lower_margin_min": float(np.min(lower_margin)),
        "limit_upper_margin_min": float(np.min(upper_margin)),
    }


def diagnose(max_waypoints: int, retries: int, output_csv: str) -> dict:
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    limits = joint_limits(model, mujoco)
    time_values, _, positions, _, camera_poses, look_targets = generate_segmented_reference(
        max_waypoints=max_waypoints,
        return_targets=True,
    )
    tcp_poses = camera_poses_to_tcp_poses(camera_poses)
    q_hist, flags = solve_trajectory(
        model,
        data,
        mujoco,
        tcp_poses,
        look_target=look_targets,
        q_start=ROBOT_READY,
        retries=retries,
        verbose=False,
        axis_col=EE_LOOK_AXIS_COL,
        axis_sign=EE_LOOK_AXIS_SIGN,
        site_name=CAMERA_SITE_NAME,
    )

    qdot = np.gradient(q_hist, time_values, axis=0)
    step_dq = np.zeros_like(q_hist)
    step_dq[1:] = q_hist[1:] - q_hist[:-1]

    rows = []
    for i, (t, q, qd, dq, ok) in enumerate(zip(time_values, q_hist, qdot, step_dq, flags)):
        metrics = jacobian_metrics(tcp_jacobian(model, data, q))
        limits_info = limit_metrics(q, limits)
        row = {
            "index": i,
            "time": float(t),
            "ik_success": int(ok),
            "dq_step_norm": float(np.linalg.norm(dq)),
            "qdot_norm": float(np.linalg.norm(qd)),
            "qdot_abs_max": float(np.max(np.abs(qd))),
            **metrics,
            **limits_info,
        }
        row.update({f"q{j+1}": float(q[j]) for j in range(q_hist.shape[1])})
        row.update({f"qdot{j+1}": float(qd[j]) for j in range(q_hist.shape[1])})
        row.update({f"dq_step{j+1}": float(dq[j]) for j in range(q_hist.shape[1])})
        rows.append(row)

    fieldnames = list(rows[0].keys())
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "rows": rows,
        "flags": flags,
        "q_hist": q_hist,
        "time": time_values,
        "positions": positions,
        "csv": output_csv,
    }


def print_report(result: dict, top_k: int) -> None:
    rows = result["rows"]
    print(f"[diagnostics] wrote CSV: {result['csv']}")
    print(f"[diagnostics] IK success: {sum(result['flags'])}/{len(result['flags'])}")

    for title, key, reverse in [
        ("largest joint step", "dq_step_norm", True),
        ("largest joint velocity", "qdot_norm", True),
        ("worst condition", "condition", True),
        ("smallest min singular value", "min_sv", False),
        ("smallest manipulability", "manipulability", False),
    ]:
        print(f"\n[{title}]")
        ranked = sorted(rows, key=lambda r: r[key], reverse=reverse)[:top_k]
        for r in ranked:
            q = [r[f"q{j+1}"] for j in range(6)]
            qdot = [r[f"qdot{j+1}"] for j in range(6)]
            print(
                f"idx={r['index']:3d} t={r['time']:7.3f} "
                f"ok={r['ik_success']} dq={r['dq_step_norm']:.4f} "
                f"|qdot|={r['qdot_norm']:.4f} max|qdot|={r['qdot_abs_max']:.4f} "
                f"cond={r['condition']:.2f} min_sv={r['min_sv']:.5f} "
                f"manip={r['manipulability']:.6g} "
                f"limit=q{int(r['limit_closest_joint'])} margin={r['limit_margin_min']:.4f}"
            )
            print("  q    =", np.array2string(np.asarray(q), precision=4, suppress_small=True))
            print("  qdot =", np.array2string(np.asarray(qdot), precision=4, suppress_small=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-waypoints", type=int, default=240)
    parser.add_argument("--retries", type=int, default=16)
    parser.add_argument("--csv", default="outputs/diagnostics/trajectory_diagnostics.csv")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    result = diagnose(args.max_waypoints, args.retries, args.csv)
    print_report(result, args.top_k)


if __name__ == "__main__":
    main()
