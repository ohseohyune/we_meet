"""
전체 파이프라인 실행 스크립트

  1단계: MuJoCo 시뮬레이션 → 뎁스 프레임 + metadata.csv 생성
  2단계: 뎁스 프레임 + 카메라 pose → 3D 형상 복원 (point cloud + mesh)
  3단계: 복원된 포인트 클라우드 → seam 추출 → 용접 경로 생성
  4단계: (선택) MuJoCo 뷰어에서 로봇이 용접 경로 시뮬레이션

사용법:
  python master.py                       # 전체 파이프라인 (1~3단계)
  python master.py --skip-capture        # 1단계 건너뛰고 복원+용접 경로만
  python master.py --skip-recon          # 1단계(뎁스 생성)만
  python master.py --skip-weld           # 1·2단계만 (용접 경로 생성 건너뜀)
  python master.py --visualize           # 3단계 완료 후 Open3D 뷰어
  python master.py --weld-viewer         # 3단계 완료 후 MuJoCo 용접 시뮬레이션
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def _mjpython() -> str:
    """Return mjpython path (needed for MuJoCo viewer on macOS), else sys.executable."""
    import shutil
    candidate = shutil.which("mjpython")
    if candidate:
        return candidate
    # Common venv install location
    venv_bin = os.path.join(os.path.dirname(sys.executable), "mjpython")
    if os.path.isfile(venv_bin):
        return venv_bin
    return sys.executable


CAPTURE_CMD = [
    sys.executable,
    os.path.join(ROOT, "mujoco_viewer.py"),
    "--ik",
    "--camera",
    "--export-csv",
    "--no-viewer",
]

RECON_CMD = [
    sys.executable,
    os.path.join(ROOT, "3D_reconstruction", "Reconstruct_3D.py"),
]

PLY_PATH   = os.path.join(ROOT, "inspection_frames", "output", "reconstructed_pointcloud.ply")
WELD_CSV   = os.path.join(ROOT, "inspection_frames", "output", "weld_trajectory.csv")


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {step}")
    print(f"  {' '.join(os.path.basename(c) if i > 0 else c for i, c in enumerate(cmd))}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n[pipeline] {step} 실패 (exit code {result.returncode}). 중단합니다.")
        sys.exit(result.returncode)


def run_weld_planning(visualize: bool = False) -> None:
    """3단계: seam 추출 → 용접 경로 생성 (in-process)."""
    print(f"\n{'='*60}")
    print("  3단계: seam 추출 → 용접 경로 생성")
    print(f"{'='*60}\n")

    sys.path.insert(0, ROOT)
    from welding.seam_extraction import extract_seam
    from welding.weld_trajectory import generate_weld_poses, save_weld_trajectory_csv

    seam = extract_seam(PLY_PATH, verbose=True)
    poses, angles = generate_weld_poses(
        center=seam["center"],
        radius=seam["radius"],
        normal=seam["normal"],
    )
    save_weld_trajectory_csv(WELD_CSV, poses, angles)

    if visualize:
        from welding.visualize import show_weld_result
        show_weld_result(PLY_PATH, seam, poses, angles)


def main() -> None:
    parser = argparse.ArgumentParser(description="촬영 → 3D 복원 → 용접 경로 전체 파이프라인")
    parser.add_argument("--skip-capture", action="store_true",
                        help="1단계(뎁스 생성) 건너뛰기 — inspection_frames/가 이미 있을 때")
    parser.add_argument("--skip-recon", action="store_true",
                        help="2단계(3D 복원) 건너뛰기")
    parser.add_argument("--skip-weld", action="store_true",
                        help="3단계(용접 경로 생성) 건너뛰기")
    parser.add_argument("--visualize", action="store_true",
                        help="3단계 완료 후 Open3D 뷰어로 용접 경로 확인")
    parser.add_argument("--weld-viewer", action="store_true",
                        help="3단계 완료 후 MuJoCo 뷰어에서 로봇 용접 시뮬레이션")
    parser.add_argument("--recon-args", nargs=argparse.REMAINDER, default=[],
                        help="Reconstruct_3D.py에 추가로 전달할 인자")
    args = parser.parse_args()

    if not args.skip_capture:
        run(CAPTURE_CMD, "1단계: 시뮬레이션 뎁스 캡처 + CSV 생성")
    else:
        print("[pipeline] 1단계 건너뜀 — 기존 inspection_frames/ 사용")

    if not args.skip_recon:
        recon_cmd = RECON_CMD + args.recon_args
        run(recon_cmd, "2단계: 3D 형상 복원")
    else:
        print("[pipeline] 2단계 건너뜀")

    if not args.skip_weld:
        run_weld_planning(visualize=args.visualize)
    else:
        print("[pipeline] 3단계 건너뜀")

    if args.weld_viewer:
        weld_viewer_cmd = [
            _mjpython(),
            os.path.join(ROOT, "mujoco_viewer.py"),
            "--weld",
        ]
        run(weld_viewer_cmd, "4단계: MuJoCo 용접 시뮬레이션")

    print("\n[pipeline] 완료.")
    print(f"  포인트 클라우드: {PLY_PATH}")
    print(f"  메쉬:            {os.path.join(ROOT, 'inspection_frames', 'output', 'reconstructed_mesh.ply')}")
    print(f"  용접 경로:       {WELD_CSV}")


if __name__ == "__main__":
    main()
