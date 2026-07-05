"""
visualize.py — 용접 경로 Open3D 시각화.

표시 내용
---------
  - 복원된 포인트 클라우드 (Z 높이 기준 색상)
  - 추출된 seam 원 (빨간 선)
  - 용접 waypoint 좌표계 (X=초록/진행, Z=파랑/토치축)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _color_by_z(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """포인트 클라우드를 Z 높이 기준으로 viridis 유사 색상으로 칠한다."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    z = pts[:, 2]
    t = (z - z.min()) / (z.max() - z.min() + 1e-9)   # 0~1 정규화

    # viridis 근사: purple(0) → teal(0.5) → yellow(1)
    r = np.clip(1.8 * t - 0.4,      0.0, 1.0)
    g = np.clip(1.8 * t - 0.2,      0.0, 1.0) * np.clip(1.8 * (1 - t) + 0.2, 0.0, 1.0)
    b = np.clip(1.5 * (1 - t) - 0.1, 0.0, 1.0)

    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(np.column_stack([r, g, b]))
    return out


def _seam_circle_tube(
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
    n: int = 200,
    tube_radius: float = 0.0020,   # 선 굵기 (m)
    color: list = None,
) -> o3d.geometry.TriangleMesh:
    """
    seam 원을 tube 메쉬로 만든다.
    LineSet은 굵기 조절이 불가능하므로 구(sphere) 체인으로 표현한다.
    tube_radius = 선 반지름 (기본 2 mm).
    """
    if color is None:
        color = [1.0, 0.1, 0.1]   # 빨강

    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([0.0, 0.0, 1.0])
    u = ref - np.dot(ref, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    phis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    seam_pts = [center + radius * (np.cos(p) * u + np.sin(p) * v) for p in phis]

    combined = o3d.geometry.TriangleMesh()
    for pt in seam_pts:
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=tube_radius, resolution=8
        )
        sphere.translate(pt)
        combined += sphere

    combined.paint_uniform_color(color)
    combined.compute_vertex_normals()
    return combined


def _pose_frames_lineset(
    poses: list[np.ndarray],
    scale: float = 0.008,
    every: int = 1,
) -> o3d.geometry.LineSet:
    """
    각 EE pose에 작은 좌표축 선분을 추가한다.
      X (초록) = 진행 방향
      Y (회색)
      Z (파랑) = 토치 축
    """
    axis_colors = [
        [0.1, 0.9, 0.1],   # X — 초록 (진행 방향)
        [0.6, 0.6, 0.6],   # Y — 회색
        [0.1, 0.4, 1.0],   # Z — 파랑 (토치 축)
    ]

    points, lines, colors = [], [], []
    idx = 0
    for T in poses[::every]:
        pos = T[:3, 3]
        for j in range(3):
            end = pos + T[:3, j] * scale
            points.extend([pos, end])
            lines.append([idx, idx + 1])
            colors.append(axis_colors[j])
            idx += 2

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _center_sphere(center: np.ndarray, r: float = 0.003) -> o3d.geometry.TriangleMesh:
    """seam 중심에 작은 구를 표시한다."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
    sphere.translate(center)
    sphere.paint_uniform_color([1.0, 0.5, 0.0])   # 주황
    sphere.compute_vertex_normals()
    return sphere


# ── 공개 API ──────────────────────────────────────────────────────────────────

def show_weld_result(
    ply_path: str | Path,
    seam: dict,
    poses: list[np.ndarray],
    angles: np.ndarray,
    frame_scale: float = 0.008,
    show_every: int = 3,
) -> None:
    """
    복원 포인트 클라우드 + seam 원 + 용접 waypoint 좌표계를 Open3D로 표시.

    Parameters
    ----------
    ply_path    : 복원된 PLY 경로
    seam        : extract_seam() 반환값 (center, radius, normal)
    poses       : generate_weld_poses() 반환 4×4 변환 리스트
    angles      : generate_weld_poses() 반환 각도 배열
    frame_scale : 좌표축 선분 길이 (m)
    show_every  : waypoint 중 n개 간격으로만 표시 (화면 과밀 방지)
    """
    print("[viz] Open3D 뷰어 로딩 중...")

    # 포인트 클라우드
    pcd_raw = o3d.io.read_point_cloud(str(ply_path))
    pcd = _color_by_z(pcd_raw)

    # seam 원 (tube) + 중심
    seam_line = _seam_circle_tube(seam["center"], seam["radius"], seam["normal"])
    seam_ctr  = _center_sphere(seam["center"])

    # 용접 waypoint 좌표계
    frame_ls = _pose_frames_lineset(poses, scale=frame_scale, every=show_every)

    # 범례 출력
    n_shown = len(poses[::show_every])
    print(f"[viz] 포인트 클라우드: {len(pcd.points):,}점")
    print(f"[viz] seam center : {seam['center']}")
    print(f"[viz] seam radius : {seam['radius']*1000:.2f} mm")
    print(f"[viz] 용접 waypoints: {len(poses)}개  ({n_shown}개 표시)")
    print("[viz] 축 색상 — 초록: 진행방향(X)  파랑: 토치축(Z)  빨강: seam 원")
    print("[viz] 뷰어 창을 닫으면 종료됩니다.")

    o3d.visualization.draw_geometries(
        [pcd, seam_line, seam_ctr, frame_ls],
        window_name="Weld Trajectory Preview",
        width=1280,
        height=800,
        point_show_normal=False,
    )
