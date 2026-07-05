"""
seam_extraction.py — 복원된 포인트 클라우드에서 파이프-플랜지 용접 심(seam) 추출.

파이프 OD가 플랜지 면과 만나는 원형 접합선(seam)을 RANSAC으로 fitting해
center / radius / normal 을 반환한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── ROI 기본값 (scene.xml 실측 기하 기준) ────────────────────────────────────
# scene.xml: body pos=[0.565, 0, 0.570], euler="0 1.5708 0" (Y축 90° 회전)
#   → body +Z = world +X
#   pipe_outer : center world [0.565, 0, 0.570], half_len=0.1250 → X 0.440~0.690
#   flange_disc: center world [0.703, 0, 0.570], half_len=0.0128 → X 0.690~0.716
# seam = 파이프 끝(X=0.690)이 플랜지 면과 만나는 원, r = pipe_OD/2 = 0.03025m
#
# NOTE: generator.py의 FLANGE_CENTER X=0.810은 scene.xml과 120mm 불일치하므로
#       여기서는 scene.xml 측정값을 직접 사용한다.
FLANGE_X     = 0.690    # scene.xml 기준 실제 seam X 위치 (m)
PIPE_AXIS_Y  = 0.000    # 파이프 축 Y (m)
PIPE_AXIS_Z  = 0.570    # 파이프 축 Z (m)
SEAM_R_NOM   = 0.03025  # 파이프 OD / 2 (m)


def load_point_cloud(ply_path: str | Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        raise ValueError(f"포인트 클라우드가 비어 있습니다: {ply_path}")
    return pcd


def _extract_pipe_od_points(
    pts: np.ndarray,
    pipe_axis_y: float = PIPE_AXIS_Y,
    pipe_axis_z: float = PIPE_AXIS_Z,
    seam_r_nom: float = SEAM_R_NOM,
    r_tol_factor: float = 0.35,   # seam_r_nom ± 35%
    x_max: float | None = None,   # None → pts[:,0].max()
) -> np.ndarray:
    """파이프 OD 표면 포인트 추출 (반경 조건만, X 범위 넓게)."""
    r = np.sqrt((pts[:, 1] - pipe_axis_y) ** 2 + (pts[:, 2] - pipe_axis_z) ** 2)
    r_lo = seam_r_nom * (1.0 - r_tol_factor)
    r_hi = seam_r_nom * (1.0 + r_tol_factor)
    mask = (r >= r_lo) & (r <= r_hi)
    if x_max is not None:
        mask &= pts[:, 0] <= x_max
    return pts[mask]


def _fit_circle_2d_yz(
    pts: np.ndarray,
    pipe_axis_y: float = PIPE_AXIS_Y,
    pipe_axis_z: float = PIPE_AXIS_Z,
) -> tuple[float, float, float]:
    """
    Y-Z 평면 대수 원 fitting.  파이프 축이 X임을 알고 있으므로
    PCA 법선 추정 없이 직접 2D fitting 한다.

    Returns: cy (세계 Y), cz (세계 Z), radius
    """
    y = pts[:, 1] - pipe_axis_y
    z = pts[:, 2] - pipe_axis_z
    A = np.column_stack([2 * y, 2 * z, np.ones(len(y))])
    b = y ** 2 + z ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    dy, dz, c = sol
    radius = float(np.sqrt(np.clip(c + dy ** 2 + dz ** 2, 0.0, None)))
    return float(pipe_axis_y + dy), float(pipe_axis_z + dz), radius


def fit_circle_pipe_axis(
    pts: np.ndarray,
    pipe_axis_y: float = PIPE_AXIS_Y,
    pipe_axis_z: float = PIPE_AXIS_Z,
    seam_r_nom: float = SEAM_R_NOM,
    inlier_tol: float = 0.003,
    seam_x_percentile: float = 97.0,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """
    파이프 축(X축)을 알고 있는 경우의 seam 원 fitting.

    알고리즘
    --------
    1. Y-Z 2D circle fitting으로 파이프 축 중심과 OD 반지름 추정
       (X 좌표는 무시 → 파이프 원통 표면이 -X로 분산되어도 영향 없음)
    2. seam X = OD 포인트 중 X 상위 percentile
       (파이프 끝단의 위치 = seam 평면의 X)
    3. normal = [1, 0, 0] (파이프 축 방향으로 고정)

    Returns
    -------
    center    : np.ndarray (3,)
    radius    : float
    normal    : np.ndarray (3,)
    n_inliers : int
    """
    if len(pts) < 6:
        raise ValueError(f"포인트 수 부족 (got {len(pts)}, need ≥ 6)")

    # 1. 2D circle fitting (Y-Z)
    cy, cz, radius = _fit_circle_2d_yz(pts, pipe_axis_y, pipe_axis_z)

    # 2. seam X: OD 포인트 상위 percentile (이상치에 강건)
    seam_x = float(np.percentile(pts[:, 0], seam_x_percentile))

    center = np.array([seam_x, cy, cz])
    normal = np.array([1.0, 0.0, 0.0])

    # 3. inlier 계산 (Y-Z 잔차 기준)
    r_each = np.sqrt((pts[:, 1] - cy) ** 2 + (pts[:, 2] - cz) ** 2)
    n_inliers = int(np.sum(np.abs(r_each - radius) < inlier_tol))

    return center, radius, normal, n_inliers


# ── 공개 API ──────────────────────────────────────────────────────────────────

def extract_seam(
    ply_path: str | Path,
    verbose: bool = True,
    pipe_axis_y: float = PIPE_AXIS_Y,
    pipe_axis_z: float = PIPE_AXIS_Z,
    seam_r_nom: float = SEAM_R_NOM,
    r_tol_factor: float = 0.35,
    inlier_tol: float = 0.003,
    seam_x_percentile: float = 97.0,
) -> dict:
    """
    PLY 포인트 클라우드 → seam 원 파라미터.

    파이프 축이 world X임을 이용해 Y-Z 평면 2D circle fitting으로 중심·반지름을
    구하고, seam X 위치는 파이프 OD 포인트의 상위 percentile로 결정한다.

    Returns
    -------
    {
        "center"   : np.ndarray (3,),
        "radius"   : float,
        "normal"   : np.ndarray (3,),   # [1, 0, 0] 고정
        "n_inliers": int,
    }
    """
    pcd = load_point_cloud(ply_path)
    pts = np.asarray(pcd.points)

    od_pts = _extract_pipe_od_points(
        pts,
        pipe_axis_y=pipe_axis_y,
        pipe_axis_z=pipe_axis_z,
        seam_r_nom=seam_r_nom,
        r_tol_factor=r_tol_factor,
    )
    print(f"[seam] 파이프 OD 후보: {len(od_pts)} / {len(pts)}")

    if len(od_pts) < 6:
        raise RuntimeError(
            f"파이프 OD 포인트 부족 ({len(od_pts)}개). r_tol_factor를 키워 보세요."
        )

    center, radius, normal, n_inliers = fit_circle_pipe_axis(
        od_pts,
        pipe_axis_y=pipe_axis_y,
        pipe_axis_z=pipe_axis_z,
        seam_r_nom=seam_r_nom,
        inlier_tol=inlier_tol,
        seam_x_percentile=seam_x_percentile,
    )

    if verbose:
        print(f"[seam] center  : {center}")
        print(f"[seam] radius  : {radius * 1000:.2f} mm  (nominal {seam_r_nom * 1000:.2f} mm)")
        print(f"[seam] normal  : {normal}")
        print(f"[seam] inliers : {n_inliers} / {len(od_pts)}")

    return {"center": center, "radius": radius, "normal": normal, "n_inliers": n_inliers}
