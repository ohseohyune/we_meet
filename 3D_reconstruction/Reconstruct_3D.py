"""
MuJoCo 시뮬레이션 depth 데이터 -> 3D 형상 복원 파이프라인
=================================================================

개요
----
metadata.csv 에는 프레임별로 다음 정보가 들어있다고 가정한다.
    - index               : 프레임 번호 (0 ~ N-1)
    - angle_deg           : 턴테이블/물체 회전각 (참고용, 본 파이프라인에서는 직접 쓰지 않음)
    - capture_valid       : 1이면 유효 캡처
    - camera_x/y/z        : world 좌표계에서 카메라(광학중심) 위치 [m]
    - camera_qw/qx/qy/qz  : world 좌표계 기준 카메라 자세 쿼터니언 (MuJoCo 컨벤션, wxyz 순서)
    - tcp_*, q1~q6        : 로봇 TCP pose / 조인트각 (이 파이프라인에서는 사용하지 않음.
                             camera_* 값이 곧 카메라 pose 이므로 그대로 사용)

각 프레임에는 짝이 되는 .npy 파일이 있고, 그 안에는 depth map
(H=800, W=1280, float32, 단위: meter) 이 들어있다.
카메라는 Intel RealSense D405 스펙(FOV 87° x 58°, depth range 7~50cm)을 사용한다고 가정한다.

처리 절차
--------
1. metadata.csv 로부터 각 프레임의 camera pose(world frame) 를 읽는다.
2. depth map -> camera-frame point cloud 로 역투영(unproject) 한다.
   (D405 데이터시트 기준 intrinsic 사용. 픽셀 종횡비를 (800,1280) 해상도에 맞게 재계산.)
3. depth range(7~50cm) 밖의 값(센서 클리핑/배경)은 제거한다.
4. camera pose를 이용해 point cloud를 world frame으로 변환한다.
   (MuJoCo 카메라는 로컬 -z 방향을 바라보고 로컬 +y가 "위" 이므로,
    OpenCV/Open3D가 기대하는 +z-forward, -y-down 카메라 좌표계로 축을 먼저 변환한다.)
5. world frame으로 옮긴 모든 프레임의 point cloud를 합친 뒤,
   Open3D ICP(point-to-plane)로 인접 프레임끼리 미세 정합(refine)하여
   MuJoCo pose의 미세한 오차를 보정한다.
   (camera pose 자체가 이미 정확한 ground-truth 라면 ICP 정합 없이도 합칠 수 있지만,
    여기서는 pose 노이즈를 보정하는 용도로 ICP를 pairwise refinement에 사용한다.)
6. 정합된 point cloud를 outlier 제거 + voxel downsample 하여 최종 3D 형상을 만든다.
7. (선택) Poisson/Alpha-shape 등으로 메쉬 복원까지 수행한다.

사용 방법
--------
python reconstruct_3d.py \
    --csv /path/to/metadata.csv \
    --depth_dir /path/to/npy_folder \
    --out_dir ./output \
    --depth_min 0.07 --depth_max 0.5

depth .npy 파일 이름이 "{timestamp}_frame_{index:03d}.npy" 같은 패턴이라
정확히 index와 1:1 매핑이 안 되는 경우, --depth_glob 옵션으로 매칭 패턴을 바꿀 수 있다.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# ----------------------------------------------------------------------------
# 카메라 intrinsic (MuJoCo scene.xml fovy 기반)
# ----------------------------------------------------------------------------
# MuJoCo는 수직 FOV(fovy)로 pinhole 카메라를 정의하고 정사각 픽셀(fx==fy)을 사용한다.
# scene.xml: <camera name="d405_camera" fovy="65" />
# → fy = (height/2) / tan(fovy/2), fx = fy
# D405 하드웨어 스펙(87°H / 58°V)을 그대로 쓰면 focal length가 ~8% 틀려
# 역투영 시 3D 좌표가 체계적으로 오차가 생기므로 실제 렌더링 파라미터를 따른다.
MUJOCO_FOVY_DEG = 65.0  # scene.xml fovy 값과 반드시 일치시킬 것

# 실제 depth map 해상도 (npy shape) - 필요시 커맨드라인에서 override 가능
DEFAULT_DEPTH_W = 1280
DEFAULT_DEPTH_H = 800

# D405 권장 동작 거리 (m) - 이 범위 밖의 depth 값은 신뢰도가 낮거나
# 센서 클리핑(배경)일 가능성이 높으므로 기본값으로 필터링한다.
D405_MIN_RANGE_M = 0.07
D405_MAX_RANGE_M = 0.50


def build_intrinsic(width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    """
    MuJoCo fovy 기반으로 focal length를 계산한다.
    MuJoCo는 정사각 픽셀(fx==fy)을 사용하므로 수직 FOV에서 fy를 구하고 fx=fy로 둔다.
    cx, cy는 이미지 중심 (calibration 데이터 없으므로 가장 합리적인 기본값).
    """
    fy = (height / 2.0) / np.tan(np.deg2rad(MUJOCO_FOVY_DEG / 2.0))
    fx = fy  # MuJoCo square pixel
    cx = width / 2.0
    cy = height / 2.0
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


# ----------------------------------------------------------------------------
# MuJoCo camera pose -> 4x4 homogeneous transform (world <- camera)
# ----------------------------------------------------------------------------
def mujoco_cam_pose_to_T(pos_xyz, quat_wxyz) -> np.ndarray:
    """
    MuJoCo 카메라는 로컬 좌표계에서 -z 방향을 바라보고 +y가 "위쪽" 이다.
    Open3D / OpenCV 핀홀 카메라 모델은 +z가 전방(보는 방향), +y가 "아래쪽" 이다.
    따라서 world <- mujoco_cam 변환행렬을 만든 뒤,
    mujoco_cam -> cv_cam 축 변환(diag(1,-1,-1))을 추가로 곱해
    최종적으로 world <- cv_cam 변환행렬(extrinsic의 inverse)을 얻는다.
    """
    x, y, z = pos_xyz
    qw, qx, qy, qz = quat_wxyz

    # scipy Rotation은 (x,y,z,w) 순서를 사용하므로 순서 변환
    rot_world_from_mjcam = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # mujoco-cam axes -> opencv/open3d-cam axes
    # mujoco: x-right, y-up,    z-backward(camera looks at -z)
    # cv/o3d: x-right, y-down,  z-forward
    flip = np.diag([1.0, -1.0, -1.0])

    rot_world_from_cvcam = rot_world_from_mjcam @ flip

    T_world_from_cvcam = np.eye(4)
    T_world_from_cvcam[:3, :3] = rot_world_from_cvcam
    T_world_from_cvcam[:3, 3] = [x, y, z]
    return T_world_from_cvcam


# ----------------------------------------------------------------------------
# 프레임 로딩
# ----------------------------------------------------------------------------
def find_depth_file(depth_dir: str, index: int, depth_glob: str | None) -> str | None:
    """
    depth_dir 안에서 해당 index에 매칭되는 .npy 파일 경로를 찾는다.
    파일명 패턴이 "{어떤 prefix}_frame_{index:03d}.npy" 형태인 경우를 기본으로 처리하고,
    --depth_glob 옵션으로 패턴을 override 할 수 있다 ('{index}', '{index:03d}' 와 같은
    포맷 문자열 placeholder 지원).
    """
    if depth_glob:
        pattern = depth_glob.format(index=index)
        matches = sorted(glob.glob(os.path.join(depth_dir, pattern)))
        return matches[0] if matches else None

    # 기본 후보 패턴들을 순서대로 시도
    candidates = [
        f"*frame_{index:03d}.npy",
        f"*_frame_{index}.npy",
        f"frame_{index:03d}.npy",
        f"frame_{index}.npy",
        f"{index:03d}.npy",
        f"{index}.npy",
        f"*_{index:03d}.npy",
    ]
    for pat in candidates:
        matches = sorted(glob.glob(os.path.join(depth_dir, pat)))
        if matches:
            return matches[0]
    return None


def depth_to_pointcloud_cam(
    depth: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    depth_min: float,
    depth_max: float,
    T_world_from_cvcam: np.ndarray | None = None,
    roi_center: tuple | None = None,
    roi_radius: float | None = None,
    roi_bbox_min: tuple | None = None,
    roi_bbox_max: tuple | None = None,
) -> np.ndarray:
    """
    Depth map(H,W, meter 단위) -> camera-frame point cloud (N,3).
    depth_min/depth_max 범위 밖 픽셀(센서 동작범위 밖)은 먼저 제거한다.

    카메라가 회전하며 물체를 비스듬히 내려다보게 되면, 물체 뒤쪽의 배경/바닥이나
    물체 옆에 있는 다른 구조물(예: 받침대/스탠드)이 depth_max 안쪽으로 들어와
    같이 찍히는 경우가 있다. 이런 것들은 depth_min/depth_max만으로는 걸러낼 수
    없으므로, world frame 기준 ROI로 한 번 더 필터링한다.

    ROI는 두 가지 방식을 지원한다 (둘 다 주어지면 둘 다 적용/AND 조건):
    1) roi_center/roi_radius: 구형 ROI. 물체가 어떤 점 주위 반경 이내에 있다고 가정.
    2) roi_bbox_min/roi_bbox_max: 축정렬 박스 ROI. 예를 들어 파이프처럼 한쪽으로 긴
       물체는 구형보다 박스(특히 길이 방향 범위 제한)가 옆의 다른 물체(받침대 등)를
       훨씬 깔끔하게 잘라낼 수 있다.
    """
    assert depth.shape == (intrinsic.height, intrinsic.width), (
        f"depth shape {depth.shape} != intrinsic ({intrinsic.height}, {intrinsic.width})"
    )
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()

    valid = (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
    vs, us = np.nonzero(valid)
    zs = depth[vs, us]

    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy

    pts_cam = np.stack([xs, ys, zs], axis=1).astype(np.float64)

    if roi_center is not None and roi_radius is None:
        print("[depth_to_pointcloud_cam] 경고: roi_center가 지정됐지만 roi_radius=None 이어서 "
              "구형 ROI 필터가 적용되지 않습니다.", file=sys.stderr)

    if T_world_from_cvcam is not None and (roi_center is not None or roi_bbox_min is not None):
        pts_world = (T_world_from_cvcam[:3, :3] @ pts_cam.T + T_world_from_cvcam[:3, 3:4]).T
        keep = np.ones(len(pts_world), dtype=bool)

        if roi_center is not None and roi_radius is not None:
            dist = np.linalg.norm(pts_world - np.asarray(roi_center), axis=1)
            keep &= dist <= roi_radius

        if roi_bbox_min is not None and roi_bbox_max is not None:
            bbox_min = np.asarray(roi_bbox_min)
            bbox_max = np.asarray(roi_bbox_max)
            keep &= np.all((pts_world >= bbox_min) & (pts_world <= bbox_max), axis=1)

        pts_cam = pts_cam[keep]

    return pts_cam


def load_frames(
    csv_path: str,
    depth_dir: str,
    depth_glob: str | None,
    depth_min: float,
    depth_max: float,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    max_frames: int | None = None,
    frame_stride: int = 1,
    index_min: int | None = None,
    index_max: int | None = None,
    roi_center: tuple | None = None,
    roi_radius: float | None = None,
    roi_bbox_min: tuple | None = None,
    roi_bbox_max: tuple | None = None,
):
    """
    metadata.csv 의 모든 유효(capture_valid==1) 프레임에 대해
    (camera-frame point cloud, world<-cvcam 변환행렬) 쌍의 리스트를 만든다.

    index_min/index_max: metadata.csv의 'index' 컬럼 기준으로 정합에 사용할
    프레임 범위를 [index_min, index_max] (양끝 포함) 으로 제한한다.
    예: index_min=0, index_max=59 -> 0번부터 59번까지 60개 프레임만 사용.

    roi_center/roi_radius: world frame 기준 관심영역(구) 필터.
    roi_bbox_min/roi_bbox_max: world frame 기준 축정렬 박스 ROI 필터. 파이프처럼
    한쪽으로 긴 물체 옆에 받침대 등 다른 구조물이 있을 때, 구형 ROI보다 박스로
    길이 방향 범위를 직접 제한하는 편이 다른 물체를 더 깔끔하게 제거할 수 있다.
    """
    df = pd.read_csv(csv_path)

    if index_min is not None:
        df = df[df["index"] >= index_min]
    if index_max is not None:
        df = df[df["index"] <= index_max]

    df = df[df["capture_valid"] == 1].reset_index(drop=True)

    if frame_stride > 1:
        df = df.iloc[::frame_stride].reset_index(drop=True)
    if max_frames is not None:
        df = df.iloc[:max_frames].reset_index(drop=True)

    frames = []
    skipped = []
    for _, row in df.iterrows():
        idx = int(row["index"])
        depth_path = find_depth_file(depth_dir, idx, depth_glob)
        if depth_path is None:
            skipped.append(idx)
            continue

        depth = np.load(depth_path).astype(np.float64)
        if depth.shape != (intrinsic.height, intrinsic.width):
            # depth map 해상도가 intrinsic과 다르면 그 프레임에 맞춰 intrinsic을 재계산
            frame_intrinsic = build_intrinsic(depth.shape[1], depth.shape[0])
        else:
            frame_intrinsic = intrinsic

        T_world_from_cvcam = mujoco_cam_pose_to_T(
            (row["camera_x"], row["camera_y"], row["camera_z"]),
            (row["camera_qw"], row["camera_qx"], row["camera_qy"], row["camera_qz"]),
        )

        pts_cam = depth_to_pointcloud_cam(
            depth, frame_intrinsic, depth_min, depth_max,
            T_world_from_cvcam=T_world_from_cvcam,
            roi_center=roi_center,
            roi_radius=roi_radius,
            roi_bbox_min=roi_bbox_min,
            roi_bbox_max=roi_bbox_max,
        )
        if pts_cam.shape[0] == 0:
            skipped.append(idx)
            continue

        frames.append(
            {
                "index": idx,
                "angle_deg": float(row["angle_deg"]),
                "depth_path": depth_path,
                "pts_cam": pts_cam,
                "T": T_world_from_cvcam,
            }
        )

    if skipped:
        print(f"[load_frames] depth 파일을 찾지 못했거나 유효 포인트가 없어 건너뛴 프레임: "
              f"{len(skipped)}개 (예: {skipped[:10]})", file=sys.stderr)

    return frames


# ----------------------------------------------------------------------------
# Pairwise ICP refinement + 누적 정합
# ----------------------------------------------------------------------------
def pcd_from_points(points: np.ndarray, voxel_size: float | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3 if voxel_size else 0.01, max_nn=30)
    )
    return pcd


def refine_pose_with_icp(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    init_T: np.ndarray,
    max_correspondence_distance: float,
):
    """
    init_T(소스 카메라 pose로부터 얻은 world transform 초기값)를 시작점으로
    Open3D point-to-plane ICP를 돌려 source_pcd를 target_pcd에 더 정밀하게 정합한다.
    반환값: (transformation, fitness, inlier_rmse). 호출하는 쪽에서 fitness/rmse를
    보고 이 결과를 적용할지 말지 판단해야 한다 (포인트가 너무 적거나 형태가 안 맞으면
    ICP가 엉뚱한 변환으로 발산할 수 있기 때문).
    """
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    return result.transformation, result.fitness, result.inlier_rmse


def reconstruct(
    frames: list,
    voxel_size: float = 0.002,
    icp_max_corr_dist: float = 0.01,
    use_icp_refinement: bool = True,
    icp_stride: int = 1,
    icp_min_fitness: float = 0.3,
    icp_max_translation: float = 0.01,
    icp_min_points: int = 200,
) -> o3d.geometry.PointCloud:
    """
    모든 프레임을 world frame point cloud로 합친다.

    use_icp_refinement=True 이면, 각 프레임을 "이전까지 누적된 world point cloud"에
    ICP(point-to-plane)로 정합해 MuJoCo pose 자체의 미세한 오차를 보정한 뒤 합친다.
    (camera pose가 시뮬레이션 ground-truth라 매우 정확하다면 use_icp_refinement=False로
     설정해도 무방하다 - 그냥 pose 그대로 변환해서 합치기만 한다.)

    ICP는 점 개수가 너무 적거나(예: ROI를 벗어나 점이 얼마 안 남은 프레임) 형태가
    잘 안 맞으면 엉뚱한 변환으로 "발산"해서 점들을 멀리 날려버릴 수 있다. 이를 막기
    위해 다음 안전장치를 둔다:
    - 소스 포인트가 icp_min_points개 미만이면 ICP를 건너뛰고 pose만 사용.
    - ICP 결과의 fitness(정합된 비율)가 icp_min_fitness 미만이면 결과를 버리고 pose만 사용.
    - ICP가 찾은 translation 크기가 icp_max_translation[m]을 넘으면 (시뮬레이션 pose는
      원래 정확해야 하므로 그렇게 큰 보정이 필요할 리 없음) 비정상으로 간주해 버린다.
    """
    if not frames:
        raise RuntimeError("유효한 프레임이 하나도 없습니다.")

    # 1) 첫 프레임을 world map의 기준(reference)으로 사용
    first = frames[0]
    world_pcd = pcd_from_points(
        (first["T"][:3, :3] @ first["pts_cam"].T + first["T"][:3, 3:4]).T,
        voxel_size=voxel_size,
    )

    accumulated_points = [np.asarray(world_pcd.points)]
    n_icp_rejected = 0

    for i, fr in enumerate(frames[1:], start=1):
        # camera pose로부터 얻은 초기 world 변환
        pts_world_init = (fr["T"][:3, :3] @ fr["pts_cam"].T + fr["T"][:3, 3:4]).T
        src_pcd = pcd_from_points(pts_world_init, voxel_size=voxel_size)

        if use_icp_refinement and ((i - 1) % icp_stride == 0) and len(src_pcd.points) >= icp_min_points:
            # 누적된 world point cloud(타깃)에 맞춰 미세 정합.
            # init_T = identity: src_pcd가 이미 pose 기반으로 world frame에 놓여 있으므로,
            # ICP는 "world frame 좌표 위에서 미세 보정"만 수행한다.
            refine_T, fitness, rmse = refine_pose_with_icp(
                src_pcd,
                world_pcd,
                np.eye(4),
                max_correspondence_distance=icp_max_corr_dist,
            )
            translation_norm = np.linalg.norm(refine_T[:3, 3])

            if fitness >= icp_min_fitness and translation_norm <= icp_max_translation:
                src_pcd.transform(refine_T)
            else:
                # ICP 결과가 신뢰할 수 없음(발산 의심) -> pose 기반 변환만 사용하고 버림
                n_icp_rejected += 1

        accumulated_points.append(np.asarray(src_pcd.points))

        # 메모리 관리를 위해 일정 주기로 누적 point cloud를 다시 빌드 + 다운샘플
        if i % 20 == 0 or i == len(frames) - 1:
            merged = np.concatenate(accumulated_points, axis=0)
            world_pcd = pcd_from_points(merged, voxel_size=voxel_size)
            accumulated_points = [np.asarray(world_pcd.points)]
            print(f"  [reconstruct] {i + 1}/{len(frames)} 프레임 처리, "
                  f"누적 포인트 수: {len(world_pcd.points)}")

    if n_icp_rejected > 0:
        print(f"[reconstruct] ICP 결과가 신뢰도 기준 미달이라 무시하고 pose만 사용한 프레임: "
              f"{n_icp_rejected}개")

    return world_pcd


def remove_outliers(pcd: o3d.geometry.PointCloud, nb_neighbors=20, std_ratio=2.0) -> o3d.geometry.PointCloud:
    cleaned, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cleaned


def diagnose_coverage(pcd: o3d.geometry.PointCloud, roi_center: np.ndarray, n_bins: int = 12):
    """
    point cloud가 roi_center를 둘러싼 방향(위도/경도)별로 얼마나 고르게 덮여 있는지 진단한다.
    특정 방향의 점 개수가 다른 방향들에 비해 크게 적으면, 그 방향은 self-occlusion
    (카메라가 충분히 그 각도에서 보지 못함) 때문에 데이터가 부족하다는 뜻이고,
    Poisson reconstruction이 그 영역에서 표면을 왜곡되게 외삽할 위험이 크다는 신호다.
    """
    pts = np.asarray(pcd.points) - roi_center
    r = np.linalg.norm(pts, axis=1)
    r[r == 0] = 1e-9
    azimuth = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))   # -180~180
    elevation = np.degrees(np.arcsin(np.clip(pts[:, 2] / r, -1, 1)))  # -90~90

    az_bins = np.linspace(-180, 180, n_bins + 1)
    az_counts, _ = np.histogram(azimuth, bins=az_bins)
    az_mean = az_counts.mean()
    az_sparse = np.where(az_counts < az_mean * 0.3)[0]

    print(f"[diagnose_coverage] azimuth(경도) 방향 점 개수 분포 (총 {n_bins}개 구간, 평균 {az_mean:.0f}개):")
    print("  " + " ".join(f"{c:>5d}" for c in az_counts))
    if len(az_sparse) > 0:
        sparse_ranges = [(az_bins[i], az_bins[i + 1]) for i in az_sparse]
        print(f"  [경고] 점 밀도가 평균의 30% 미만인 azimuth 구간이 {len(az_sparse)}개 있습니다: "
              f"{sparse_ranges}")
        print("  -> 해당 각도에서 찍은 프레임이 더 있다면 --index_min/--index_max로 포함시키거나,")
        print("     mesh에서 그 영역에 왜곡/구멍이 생기는 것을 감안해서 해석해야 합니다.")
    else:
        print("  [정상] azimuth 모든 방향에서 비교적 고른 점 밀도가 확인됩니다.")

    el_bins = np.linspace(-90, 90, n_bins + 1)
    el_counts, _ = np.histogram(elevation, bins=el_bins)
    el_mean = el_counts.mean()
    el_sparse = np.where(el_counts < el_mean * 0.3)[0]

    print(f"[diagnose_coverage] elevation(위도) 방향 점 개수 분포 (총 {n_bins}개 구간, 평균 {el_mean:.0f}개):")
    print("  " + " ".join(f"{c:>5d}" for c in el_counts))
    if len(el_sparse) > 0:
        el_sparse_ranges = [(el_bins[i], el_bins[i + 1]) for i in el_sparse]
        print(f"  [경고] 점 밀도가 평균의 30% 미만인 elevation 구간이 {len(el_sparse)}개 있습니다: "
              f"{el_sparse_ranges}")
        print("  -> 물체의 상단/하단 방향 커버리지가 부족합니다. Poisson 복원에서 해당 면이 왜곡될 수 있습니다.")
    else:
        print("  [정상] elevation 모든 방향에서 비교적 고른 점 밀도가 확인됩니다.")


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
# VS Code에서 인자 없이 "실행(Run)" 버튼만 눌러도 동작하도록,
# 아래 두 경로에 본인 PC의 실제 경로를 채워넣으면 된다.
# (커맨드라인에서 --csv, --depth_dir 인자를 직접 주면 이 기본값들은 무시되고
#  그 인자값이 우선 사용된다.)
# ----------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
DEFAULT_CSV_PATH  = os.path.join(_REPO_ROOT, "inspection_frames", "metadata.csv")
DEFAULT_DEPTH_DIR = os.path.join(_REPO_ROOT, "inspection_frames", "depth_meters")
DEFAULT_OUT_DIR   = os.path.join(_REPO_ROOT, "inspection_frames", "output")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo depth 시퀀스 -> 3D 형상 복원 (Open3D ICP 활용)")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="metadata.csv 경로")
    parser.add_argument("--depth_dir", default=DEFAULT_DEPTH_DIR, help=".npy depth 파일들이 있는 폴더")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="결과물(ply 등) 저장 폴더")
    parser.add_argument("--depth_glob", default=None,
                         help="depth 파일명 매칭 패턴, 예: '*_frame_{index:03d}.npy'. "
                              "지정하지 않으면 흔한 패턴들을 자동으로 시도한다.")
    parser.add_argument("--depth_min", type=float, default=D405_MIN_RANGE_M,
                         help="유효 depth 최소값 [m] (기본: D405 최소 동작거리 0.07m)")
    parser.add_argument("--depth_max", type=float, default=D405_MAX_RANGE_M,
                         help="유효 depth 최대값 [m] (기본: D405 최대 동작거리 0.50m)")
    parser.add_argument("--voxel_size", type=float, default=0.002,
                         help="다운샘플 voxel 크기 [m] (기본 2mm)")
    parser.add_argument("--icp_max_corr_dist", type=float, default=0.01,
                         help="ICP correspondence 최대 거리 [m]")
    parser.add_argument("--use_icp", dest="no_icp", action="store_false", default=True,
                         help="ICP refinement를 켠다. 기본으로는 꺼져 있음(시뮬레이션 pose가 "
                              "이미 ground-truth 수준으로 정확하고, 점이 적은 프레임에서 "
                              "ICP가 발산해 결과를 망가뜨리는 사례가 반복 확인됨). "
                              "안전장치(fitness/translation 임계값)가 있지만 기본적으로는 비권장.")
    parser.add_argument("--icp_stride", type=int, default=1,
                         help="몇 프레임마다 ICP 정합을 수행할지 (1=매 프레임)")
    parser.add_argument("--icp_min_fitness", type=float, default=0.3,
                         help="ICP 결과를 신뢰하기 위한 최소 fitness(정합 비율, 0~1). "
                              "이보다 낮으면 ICP 결과를 버리고 camera pose만 사용한다. "
                              "(ICP가 점이 적거나 형태가 안 맞을 때 엉뚱한 변환으로 "
                              "발산하는 것을 막기 위한 안전장치)")
    parser.add_argument("--icp_max_translation", type=float, default=0.01,
                         help="ICP가 찾은 translation 보정량의 최대 허용치 [m]. 시뮬레이션 "
                              "camera pose는 원래 정확하므로 이보다 큰 보정이 나오면 ICP가 "
                              "발산한 것으로 간주해 버린다. 기본 0.01m(1cm).")
    parser.add_argument("--icp_min_points", type=int, default=200,
                         help="ICP를 시도하기 위한 프레임의 최소 포인트 개수. 이보다 적으면 "
                              "ICP를 건너뛰고 camera pose만 사용한다 (점이 너무 적으면 ICP가 "
                              "불안정해지기 때문).")
    parser.add_argument("--max_frames", type=int, default=None, help="디버깅용 - 사용할 최대 프레임 수")
    parser.add_argument("--frame_stride", type=int, default=1, help="프레임 샘플링 간격")
    parser.add_argument("--index_min", type=int, default=0,
                         help="정합에 사용할 metadata.csv 'index' 최소값 (포함). 기본 0")
    parser.add_argument("--index_max", type=int, default=215,
                         help="정합에 사용할 metadata.csv 'index' 최대값 (포함). 기본 215 "
                              "(0~215번 프레임, 전체 216개 프레임 사용)")
    parser.add_argument("--roi_center", type=float, nargs=3, default=[0.685, 0.0, 0.570],
                         metavar=("X", "Y", "Z"),
                         help="world frame 기준 관심영역(ROI) 중심 [m]. "
                              "파이프 중간점 기준 (PIPE_OFFSET_X+PIPE_LENGTH/2 = 0.685). "
                              "플랜지 face는 X=0.810, 파이프 시작은 X=0.560.")
    parser.add_argument("--roi_radius", type=float, default=0.175,
                         help="ROI 반경 [m]. roi_center(파이프 중점)로부터 이 거리 안의 점만 유지. "
                              "0.175m로 파이프 전체 길이(0.125m) + 플랜지 반경(0.09m) + 여유 포함.")
    parser.add_argument("--roi_bbox_min", type=float, nargs=3, default=[0.540, -0.110, 0.450],
                         metavar=("X", "Y", "Z"),
                         help="world frame 기준 ROI 박스의 최소 좌표 [m]. "
                              "X 최소값 0.540으로 파이프 시작(0.560) 직전 배경·받침대를 제거. "
                              "받침대가 더 잡히면 0.550~0.560까지 올릴 것.")
    parser.add_argument("--roi_bbox_max", type=float, nargs=3, default=[0.850, 0.110, 0.700],
                         metavar=("X", "Y", "Z"),
                         help="world frame 기준 ROI 박스의 최대 좌표 [m]. "
                              "X 최대값 0.850으로 플랜지 face(0.810) 너머 여유 포함.")
    parser.add_argument("--no_make_mesh", dest="make_mesh", action="store_false", default=True,
                         help="mesh 생성을 끄고 point cloud만 만든다. 기본으로는 mesh를 생성함.")
    parser.add_argument("--mesh_method", choices=["poisson", "alpha"], default="poisson",
                         help="mesh 재구성 방법. 'poisson'은 매끈하지만 데이터가 희소한 곳을 "
                              "외삽해 왜곡될 수 있고, 'alpha'는 점이 실제로 있는 곳까지만 표면을 "
                              "만들어 더 '정직'하지만 표면이 거칠고 구멍이 그대로 남을 수 있다.")
    parser.add_argument("--poisson_depth", type=int, default=9,
                         help="Poisson reconstruction의 octree depth (해상도). 기본 9. "
                              "값이 높을수록 디테일은 살지만 노이즈/왜곡도 더 잘 드러난다.")
    parser.add_argument("--poisson_density_cutoff", type=float, default=0.12,
                         help="Poisson 결과에서 밀도(신뢰도) 하위 몇 %% quantile을 제거할지 (0~1). "
                              "기본 0.12 (하위 12%% 제거). 데이터가 희소해 외삽으로 왜곡된 영역을 "
                              "줄이려면 이 값을 높인다 (예: 0.2~0.3).")
    parser.add_argument("--alpha", type=float, default=0.01,
                         help="--mesh_method alpha 사용 시 alpha 값 [m]. 점 사이 거리보다 약간 "
                              "크게 잡으면 좋다 (기본 0.01 = 1cm).")
    parser.add_argument("--depth_w", type=int, default=DEFAULT_DEPTH_W)
    parser.add_argument("--depth_h", type=int, default=DEFAULT_DEPTH_H)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    intrinsic = build_intrinsic(args.depth_w, args.depth_h)
    print(f"[main] Intrinsic (D405 기반, {args.depth_w}x{args.depth_h}): "
          f"fx=fy={intrinsic.get_focal_length()[0]:.2f}, "
          f"cx,cy={intrinsic.get_principal_point()}")

    print(f"[main] 프레임 로딩 중... (index 범위: {args.index_min} ~ {args.index_max}, "
          f"ROI 중심: {tuple(args.roi_center)}, ROI 반경: {args.roi_radius}m, "
          f"ROI 박스: {tuple(args.roi_bbox_min)} ~ {tuple(args.roi_bbox_max)})")
    frames = load_frames(
        csv_path=args.csv,
        depth_dir=args.depth_dir,
        depth_glob=args.depth_glob,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        intrinsic=intrinsic,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        index_min=args.index_min,
        index_max=args.index_max,
        roi_center=tuple(args.roi_center),
        roi_radius=args.roi_radius,
        roi_bbox_min=tuple(args.roi_bbox_min),
        roi_bbox_max=tuple(args.roi_bbox_max),
    )
    print(f"[main] 로딩된 유효 프레임 수: {len(frames)}")

    print("[main] world frame 정합 및 ICP refinement 시작...")
    pcd = reconstruct(
        frames,
        voxel_size=args.voxel_size,
        icp_max_corr_dist=args.icp_max_corr_dist,
        use_icp_refinement=not args.no_icp,
        icp_stride=args.icp_stride,
        icp_min_fitness=args.icp_min_fitness,
        icp_max_translation=args.icp_max_translation,
        icp_min_points=args.icp_min_points,
    )
    print(f"[main] 정합 완료. 다운샘플 후 포인트 수: {len(pcd.points)}")

    print("[main] outlier 제거 중...")
    pcd = remove_outliers(pcd)
    print(f"[main] outlier 제거 후 포인트 수: {len(pcd.points)}")

    # 커버리지 진단: point cloud가 ROI 구를 얼마나 고르게 덮고 있는지 확인.
    # 특정 방향에서 점 밀도가 크게 부족하면(=self-occlusion으로 못 본 영역),
    # 그 방향에서는 Poisson reconstruction이 표면을 잘못 외삽(왜곡)하기 쉽다.
    diagnose_coverage(pcd, np.asarray(args.roi_center))

    pcd_path = os.path.join(args.out_dir, "reconstructed_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"[main] point cloud 저장: {pcd_path}")

    if args.make_mesh:
        if args.mesh_method == "poisson":
            print(f"[main] Poisson surface reconstruction 중 (depth={args.poisson_depth}, "
                  f"density_quantile_cutoff={args.poisson_density_cutoff})...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.voxel_size * 5, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(30)
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=args.poisson_depth)
            densities = np.asarray(densities)
            # 밀도가 낮은(=실측 데이터가 부족해 외삽으로 채워진) vertex를 적극적으로 제거.
            # 기본 cutoff를 0.02 -> 0.12로 높여, 데이터가 희소해서 왜곡됐을 가능성이 높은
            # 영역(예: self-occlusion으로 가려졌던 면)을 더 많이 잘라낸다.
            # 그 결과 표면에 구멍이 남을 수 있지만, "잘못된 모양으로 채워진 표면"보다
            # "정직하게 비어있는 구멍"이 실제 형상 파악에는 더 안전하다.
            vertices_to_remove = densities < np.quantile(densities, args.poisson_density_cutoff)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        else:  # alpha shape - 점이 실제로 존재하는 영역에만 표면을 만들고, 빈 곳은 구멍으로 남긴다.
            print(f"[main] Alpha-shape surface reconstruction 중 (alpha={args.alpha})...")
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, args.alpha)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        mesh_path = os.path.join(args.out_dir, "reconstructed_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"[main] mesh 저장: {mesh_path} (vertices={len(mesh.vertices)}, "
              f"triangles={len(mesh.triangles)})")

    print("[main] 완료.")


if __name__ == "__main__":
    main()