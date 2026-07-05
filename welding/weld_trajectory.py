"""
weld_trajectory.py — seam 원 파라미터 → 용접 EE 경로 생성.

토치는 EE 끝단에 달린다고 가정한다.
EE 좌표계:
  +Z  = 토치 축 (전극 끝 → 용접지 방향)
  +X  = 진행 방향 (접선)
  +Y  = Z × X (우수 좌표계 완성)

용접 자세:
  - 토치 축: 플랜지 법선과 반경 방향의 45° bisector (fillet 적정각)
  - push angle: 진행 방향으로 약간 기울여 용접 풀 시야 확보
  - 위치: seam 위에서 standoff 만큼 떨어진 지점
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory.circle import is_in_bottom_forbidden_sector

# ── 기본 용접 파라미터 ────────────────────────────────────────────────────────
WELD_STANDOFF_M      = 0.012    # 토치 끝 ~ seam 거리 (12 mm)
WELD_TOOL_LENGTH_M   = 0.050    # EE origin ~ 토치 끝 거리 (50 mm)
WELD_PUSH_ANGLE_DEG  = 10.0    # 진행 방향 push angle (°)
WELD_N_WAYPOINTS     = 200     # 기본 waypoint 수 (하단 제외 후)

# 위쪽에서 용접: 파이프 하단 반원 제외 → 상단 반원(phi≈0°~180°)만 용접
# center=270°(하단), half=90° → excluded [180°, 360°], included (0°, 180°)
WELD_EXCLUDE_CENTER_RAD = np.deg2rad(270.0)
WELD_EXCLUDE_HALF_RAD   = np.deg2rad(90.0)


def _is_weld_excluded(phi: float) -> bool:
    """True when phi is in the weld-specific exclusion sector (left-bottom)."""
    diff = (phi - WELD_EXCLUDE_CENTER_RAD + np.pi) % (2 * np.pi) - np.pi
    return abs(diff) <= WELD_EXCLUDE_HALF_RAD


def _build_frame(u: np.ndarray, v: np.ndarray, normal: np.ndarray) -> tuple:
    """seam 평면의 직교 기저 (u, v)와 법선에서 phi별 단위 벡터를 계산하는 클로저 반환."""
    def at(phi: float):
        r_hat = np.cos(phi) * u + np.sin(phi) * v   # 반경 방향 (outward)
        t_hat = -np.sin(phi) * u + np.cos(phi) * v  # 접선 방향 (CCW)
        return r_hat, t_hat
    return at


def generate_weld_poses(
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
    n_waypoints: int = WELD_N_WAYPOINTS,
    standoff: float = WELD_STANDOFF_M,
    tool_length: float = WELD_TOOL_LENGTH_M,
    push_angle_deg: float = WELD_PUSH_ANGLE_DEG,
    travel_ccw: bool = True,
    exclude_bottom: bool = True,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    용접 EE 경로 생성.

    Parameters
    ----------
    center        : seam 원 중심 (3D, world frame)
    radius        : seam 반지름 (m)
    normal        : seam 평면 법선 (≈ +X, world frame)
    n_waypoints   : 원 위의 균일 샘플 수 (하단 제외 전)
    standoff      : 토치 끝 ~ seam 거리 (m)
    tool_length   : EE origin ~ 토치 끝 거리 (m); EE 위치를 추가로 후퇴시킴
    push_angle_deg: 진행 방향 push angle (°)
    travel_ccw    : CCW 방향 진행 여부

    Returns
    -------
    poses  : list of (N,) 4×4 numpy arrays — world frame EE poses
    angles : (N,) 각도 배열 (rad)
    """
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)

    # seam 평면 내 직교 기저 (u ≈ Y, v ≈ Z)
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([0.0, 0.0, 1.0])
    u = ref - np.dot(ref, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)          # normal × u → CCW 기준 v

    frame_at = _build_frame(u, v, normal)
    push_rad  = np.deg2rad(push_angle_deg)

    all_angles = np.linspace(0.0, 2.0 * np.pi, n_waypoints, endpoint=False)
    poses: list[np.ndarray] = []
    valid_angles: list[float] = []

    for phi in all_angles:
        if exclude_bottom and _is_weld_excluded(phi):
            continue

        r_hat, t_hat = frame_at(phi)
        if not travel_ccw:
            t_hat = -t_hat

        seam_pt = center + radius * r_hat

        # 토치 축: 플랜지 법선(+X≈normal)과 inward 반경(-r_hat)의 45° bisector
        # → fillet 코너를 향해 정확히 45°로 진입 (파이프 쪽에서 접근)
        z_torch = (normal - r_hat) / np.sqrt(2.0)
        z_torch /= np.linalg.norm(z_torch)

        # push angle: z_torch를 진행 방향(t_hat)으로 약간 기울임
        z_torch = np.cos(push_rad) * z_torch + np.sin(push_rad) * t_hat
        z_torch /= np.linalg.norm(z_torch)

        # EE 위치: seam에서 -z_torch 방향으로 (standoff + tool_length) 후퇴
        # standoff: 토치 끝 ~ seam 간격
        # tool_length: EE origin ~ 토치 끝 거리 (툴이 EE에 달려있는 경우)
        pos = seam_pt - (standoff + tool_length) * z_torch

        # EE 회전: x=접선, z=토치축, y=z×x
        x_ee = t_hat
        y_ee = np.cross(z_torch, x_ee)
        if np.linalg.norm(y_ee) < 1e-9:
            y_ee = np.cross(z_torch, np.array([1.0, 0.0, 0.0]))
        y_ee /= np.linalg.norm(y_ee)

        T = np.eye(4)
        T[:3, 0] = x_ee
        T[:3, 1] = y_ee
        T[:3, 2] = z_torch
        T[:3, 3] = pos
        poses.append(T)
        valid_angles.append(phi)

    return poses, np.asarray(valid_angles)


# ── CSV 저장 ──────────────────────────────────────────────────────────────────

def save_weld_trajectory_csv(
    path: str | Path,
    poses: list[np.ndarray],
    angles: np.ndarray,
) -> None:
    """용접 waypoint를 CSV로 저장 (inspection CSV와 동일 포맷 확장)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index", "phi_deg",
        "x", "y", "z",
        "qw", "qx", "qy", "qz",
    ]

    def rot_to_quat(R: np.ndarray) -> np.ndarray:
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            return np.array([0.25 / s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (T, phi) in enumerate(zip(poses, angles)):
            q = rot_to_quat(T[:3, :3])
            writer.writerow({
                "index": i,
                "phi_deg": float(np.rad2deg(phi)),
                "x": float(T[0, 3]),
                "y": float(T[1, 3]),
                "z": float(T[2, 3]),
                "qw": float(q[0]), "qx": float(q[1]),
                "qy": float(q[2]), "qz": float(q[3]),
            })

    print(f"[weld] 용접 경로 저장: {path}  ({len(poses)} waypoints)")
