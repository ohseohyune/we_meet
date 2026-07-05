"""
포인트 클라우드 / 메쉬 시각화

사용법:
  python 3D_reconstruction/visualize.py                    # 포인트 클라우드 + 인터랙티브 뷰어
  python 3D_reconstruction/visualize.py --mesh             # 메쉬 표시
  python 3D_reconstruction/visualize.py --both             # 포인트 클라우드 + 메쉬 동시 표시
  python 3D_reconstruction/visualize.py --save-png         # 인터랙티브 뷰어 없이 PNG만 저장
  python 3D_reconstruction/visualize.py --ply path/to.ply  # 파일 직접 지정

조작 방법 (Open3D 뷰어):
  마우스 드래그    : 회전
  Ctrl + 드래그   : 이동
  스크롤          : 줌
  R               : 뷰 리셋
  Q / Esc         : 종료
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
_OUT_DIR    = os.path.join(_REPO_ROOT, "inspection_frames", "output")

DEFAULT_PCD_PATH  = os.path.join(_OUT_DIR, "reconstructed_pointcloud.ply")
DEFAULT_MESH_PATH = os.path.join(_OUT_DIR, "reconstructed_mesh.ply")


# ── Open3D 뷰어 ───────────────────────────────────────────────────────────────

def view_open3d(geometries: list, title: str = "3D Reconstruction") -> None:
    import open3d as o3d
    print(f"[view] Open3D 뷰어 열기: '{title}'")
    print("       조작: 마우스 드래그=회전  Ctrl+드래그=이동  스크롤=줌  R=리셋  Q=종료")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=800,
        point_show_normal=False,
    )


# ── matplotlib 멀티뷰 PNG ────────────────────────────────────────────────────

def save_multiview_png(points: np.ndarray, out_path: str, title: str = "3D Reconstruction") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # 포인트가 너무 많으면 서브샘플 (matplotlib은 50만 점 이상에서 느림)
    MAX_PTS = 80_000
    pts = points
    if len(pts) > MAX_PTS:
        idx = np.random.choice(len(pts), MAX_PTS, replace=False)
        pts = pts[idx]
        print(f"[view] matplotlib: {len(points):,} → {MAX_PTS:,}점 랜덤 서브샘플")

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{title}  ({len(points):,} points)", fontsize=14, fontweight="bold")

    # 색상: Z 높이 기준 컬러맵
    z = pts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    colors = plt.cm.viridis(z_norm)
    s = 0.3  # 점 크기

    views = [
        (20,  -60, "Perspective"),
        (90,  -90, "Top (XY)"),
        (0,   -90, "Front (XZ)"),
        (0,     0, "Side (YZ)"),
    ]
    for i, (elev, azim, label) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=s, linewidths=0)
        ax.set_xlabel("X [m]", fontsize=8)
        ax.set_ylabel("Y [m]", fontsize=8)
        ax.set_zlabel("Z [m]", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[view] PNG 저장: {out_path}")


# ── 포인트 클라우드 로드 ──────────────────────────────────────────────────────

def load_pcd(path: str):
    import open3d as o3d
    if not os.path.exists(path):
        print(f"[view] 파일 없음: {path}", file=sys.stderr)
        sys.exit(1)
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    print(f"[view] 로드: {path}")
    print(f"       포인트 수: {len(pts):,}")
    print(f"       X 범위: {pts[:,0].min():.3f} ~ {pts[:,0].max():.3f} m")
    print(f"       Y 범위: {pts[:,1].min():.3f} ~ {pts[:,1].max():.3f} m")
    print(f"       Z 범위: {pts[:,2].min():.3f} ~ {pts[:,2].max():.3f} m")
    # 색상이 없으면 Z 높이 기준으로 컬러맵 적용
    if not pcd.has_colors():
        z = pts[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
        cmap = __import__("matplotlib.cm", fromlist=["viridis"]).viridis
        colors = cmap(z_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def load_mesh(path: str):
    import open3d as o3d
    if not os.path.exists(path):
        print(f"[view] 메쉬 파일 없음: {path}", file=sys.stderr)
        sys.exit(1)
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    print(f"[view] 메쉬 로드: {path}")
    print(f"       vertices: {len(mesh.vertices):,}  triangles: {len(mesh.triangles):,}")
    return mesh


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="포인트 클라우드 / 메쉬 시각화")
    parser.add_argument("--ply",      default=None,          help="표시할 PLY 파일 경로 (기본: reconstructed_pointcloud.ply)")
    parser.add_argument("--mesh",     action="store_true",   help="메쉬 표시 (reconstructed_mesh.ply)")
    parser.add_argument("--both",     action="store_true",   help="포인트 클라우드 + 메쉬 동시 표시")
    parser.add_argument("--save-png", action="store_true",   help="멀티뷰 PNG 저장 (인터랙티브 뷰어 없이)")
    parser.add_argument("--no-viewer",action="store_true",   help="Open3D 인터랙티브 뷰어를 열지 않음")
    parser.add_argument("--png-out",  default=None,          help="PNG 저장 경로 (기본: output/overview.png)")
    args = parser.parse_args()

    geoms = []

    if args.mesh and not args.both:
        # 메쉬만
        mesh = load_mesh(DEFAULT_MESH_PATH)
        geoms.append(mesh)
        pcd_pts = np.asarray(mesh.vertices)
        title = "Reconstructed Mesh"
    elif args.both:
        # 포인트 클라우드 + 메쉬
        pcd_path = args.ply or DEFAULT_PCD_PATH
        pcd = load_pcd(pcd_path)
        mesh = load_mesh(DEFAULT_MESH_PATH)
        geoms += [pcd, mesh]
        pcd_pts = np.asarray(pcd.points)
        title = "Point Cloud + Mesh"
    else:
        # 포인트 클라우드 (기본)
        pcd_path = args.ply or DEFAULT_PCD_PATH
        pcd = load_pcd(pcd_path)
        geoms.append(pcd)
        pcd_pts = np.asarray(pcd.points)
        title = "Reconstructed Point Cloud"

    # 월드 좌표축 표시 (원점에 XYZ 화살표)
    import open3d as o3d
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    geoms.append(axes)

    # PNG 저장
    if args.save_png or args.no_viewer:
        png_path = args.png_out or os.path.join(_OUT_DIR, "overview.png")
        save_multiview_png(pcd_pts, png_path, title=title)

    # Open3D 인터랙티브 뷰어
    if not args.no_viewer:
        view_open3d(geoms, title=title)


if __name__ == "__main__":
    main()
