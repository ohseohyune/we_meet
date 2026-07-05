"""용접 시뮬레이터 컨트롤 UI 서버.

MuJoCo 오프스크린 렌더링을 MJPEG으로 스트리밍하고, SCAN → BUILD → WELD
파이프라인을 브라우저 버튼으로 구동한다. 프론트엔드는 weld_ui.html.

사용법:
  python weld_ui.py            # http://localhost:8765 접속
  python weld_ui.py --port N

구조:
  - 메인 스레드: MuJoCo 렌더 루프 + 명령 큐 처리 (macOS GL 제약)
  - 백그라운드: ThreadingHTTPServer (/, /stream, /state, /action, /params)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import mujoco

import collections

import mujoco_viewer as mv
from control.franka_ik_solver import joint_qpos_indices, set_arm_qpos

HTML_PATH   = os.path.join(ROOT, "weld_ui.html")
PLY_PATH    = os.path.join(ROOT, "inspection_frames", "output", "reconstructed_pointcloud.ply")
META_PATH   = os.path.join(ROOT, "inspection_frames", "metadata.csv")
RECON_CMD   = [sys.executable, os.path.join(ROOT, "3D_reconstruction", "Reconstruct_3D.py")]

VIEW_W, VIEW_H = 800, 600
CAM_W, CAM_H   = 512, 384
IDLE_FPS       = 10
ANIM_FPS       = 20

RECON_VIZ_PATH  = os.path.join(ROOT, "inspection_frames", "output", "recon_viz.png")
DEPTH_PNG_DIR   = os.path.join(ROOT, "inspection_frames", "depth_png")
WELD_TRAJ_CACHE = os.path.join(ROOT, "inspection_frames", "output", "weld_traj_cache.npz")

TORCH_TIP_IN_EE = np.array([0.01384, -0.00829, -0.04733])
SEGMENT_NAMES = ["12→6 (3시)", "6→12 (3시)", "12→6 (9시)", "6→12 (9시)"]


def weld_segment_starts(phi_deg: np.ndarray) -> list[int]:
    """4개 segment의 시작 인덱스. 경계 = phi 중복(6시) + 90° 진입(12시)."""
    n = len(phi_deg)
    dups = [k + 1 for k in range(n - 1) if abs(phi_deg[k + 1] - phi_deg[k]) < 1e-9]
    if len(dups) < 2:
        q = n // 4
        return [0, q, 2 * q, 3 * q]
    lo, hi = dups[0], dups[1]
    mid = next((k for k in range(lo + 1, hi)
                if abs(phi_deg[k] - 90.0) < 0.05 and phi_deg[k - 1] < 89.0),
               (lo + hi) // 2)
    return [0, lo, mid, hi]


class SimServer:
    def __init__(self, scene_xml: str):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)
        mv.hide_trajectory_markers(self.model)

        self.renderer = mujoco.Renderer(self.model, height=VIEW_H, width=VIEW_W)
        self.cam_renderer = mujoco.Renderer(self.model, height=CAM_H, width=CAM_W)
        self.cam = mujoco.MjvCamera()
        self.pipe_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pipe_flange_assembly")
        self.supp_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pipe_supporter")
        # 정면(-y)에서 약간 비스듬히: 파이프-플랜지 seam 접합부가 옆모습으로 보이고
        # 로봇 전신도 프레임에 들어오는 앵글.
        pipe_x = float(self.model.body_pos[self.pipe_bid][0]) if self.pipe_bid >= 0 else 0.65
        self.cam.lookat[:] = [pipe_x * 0.72, 0.0, 0.52]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.5, 110.0, -22.0

        self.ee_bid  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee")
        self.tcp_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.qidx    = joint_qpos_indices(self.model, mujoco)

        # 모니터 패널용 텔레메트리: 렌더 프레임마다 (t, q1..q6, tip_err) 기록
        self.telemetry: collections.deque = collections.deque(maxlen=900)
        self.telemetry_lock = threading.Lock()
        self.t0 = time.monotonic()
        self._cur_tip_err: float | None = None

        # 3D 스켈레톤: base → link1..6 → ee → torch_tip 체인의 world 좌표
        self.skel_bids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                          for nm in ("link1", "link2", "link3", "link4", "link5", "link6", "ee")]
        self.torch_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "torch_tip")
        self.latest_skel: list = []

        self.lock  = threading.Lock()
        self.frame_cond = threading.Condition()
        self.jpegs: dict[str, bytes | None] = {"world": None, "cam": None}
        self.cmd_q: queue.Queue[str] = queue.Queue()
        self.estop = threading.Event()
        self.viz_lock = threading.Lock()

        self.state = {
            "status": "대기 중",
            "action": None,            # scan | build | weld | home | None
            "busy": False,
            "phase_done": {"scan": False, "build": False, "weld": False},
            "progress": {"i": 0, "n": 0},
            "segment": -1,
            "segment_names": SEGMENT_NAMES,
            "tip_err": {"mean": None, "max": None},
            "seam": None,
            "params": {"pipe_dist_mm": round(pipe_x * 1000, 1),
                       "speed": 1.0},
        }
        # 기존 산출물이 있으면 완료 처리 (시연 시 단계 건너뛰기용)
        if os.path.isfile(META_PATH) and os.path.isdir(os.path.join(ROOT, "inspection_frames", "depth_meters")):
            self.state["phase_done"]["scan"] = True
        if os.path.isfile(PLY_PATH):
            self.state["phase_done"]["build"] = True
        self.state["n_depth"] = self._count_depth_frames()
        self.state["recon_ready"] = os.path.isfile(PLY_PATH)

        self.weld_cache: dict[float, tuple] = {}   # pipe_dist_mm(0.5mm 해상도) -> (Q, tvals, phi_deg, seam)
        self.build_proc: subprocess.Popen | None = None
        self._load_weld_cache()

    # ── WELD 궤적 캐시 영속화 ─────────────────────────────────────────────
    def _weld_key(self) -> float:
        return round(float(self.params()["pipe_dist_mm"]) * 2) / 2

    def _save_weld_cache(self, key: float, Q, tvals, phi_deg, seam):
        np.savez(WELD_TRAJ_CACHE, dist_mm=key, Q=Q, tvals=tvals, phi_deg=phi_deg,
                 seam_center=np.asarray(seam["center"], dtype=float),
                 seam_radius=float(seam["radius"]),
                 seam_normal=np.asarray(seam["normal"], dtype=float))

    def _load_weld_cache(self):
        """서버 재시작 후에도 마지막 WELD IK 궤적을 REPLAY할 수 있게 로드."""
        if not os.path.isfile(WELD_TRAJ_CACHE):
            return
        try:
            z = np.load(WELD_TRAJ_CACHE)
            seam = {"center": z["seam_center"], "radius": float(z["seam_radius"]),
                    "normal": z["seam_normal"]}
            self.weld_cache[float(z["dist_mm"])] = (z["Q"], z["tvals"], z["phi_deg"], seam)
            print(f"[UI] 저장된 WELD 궤적 로드 (base↔pipe {float(z['dist_mm']):.1f}mm)")
        except Exception as exc:
            print(f"[UI] WELD 궤적 캐시 로드 실패: {exc}")

    # ── 상태/프레임 공유 ──────────────────────────────────────────────────
    def get_state(self) -> str:
        with self.lock:
            st = dict(self.state)
            key = round(float(st["params"]["pipe_dist_mm"]) * 2) / 2
        st["replay_ready"] = key in self.weld_cache
        return json.dumps(st)

    def set_state(self, **kw):
        with self.lock:
            for k, v in kw.items():
                self.state[k] = v

    def publish_frame(self, rgb: np.ndarray, name: str = "world"):
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=75)
        with self.frame_cond:
            self.jpegs[name] = buf.getvalue()
            self.frame_cond.notify_all()

    def render(self, cam: bool = False):
        mujoco.mj_forward(self.model, self.data)
        skel = [[0.0, 0.0, 0.02]]
        skel += [[round(float(v), 4) for v in self.data.xpos[b]] for b in self.skel_bids if b >= 0]
        if self.torch_sid >= 0:
            skel.append([round(float(v), 4) for v in self.data.site_xpos[self.torch_sid]])
        with self.telemetry_lock:
            self.telemetry.append((
                round(time.monotonic() - self.t0, 3),
                [round(float(self.data.qpos[j]), 4) for j in self.qidx],
                None if self._cur_tip_err is None else round(self._cur_tip_err * 1000, 3),
            ))
            self.latest_skel = skel
        self.renderer.update_scene(self.data, camera=self.cam)
        self.publish_frame(self.renderer.render().copy())
        if cam:
            self.cam_renderer.update_scene(self.data, camera="d405_camera")
            self.publish_frame(self.cam_renderer.render().copy(), "cam")

    @staticmethod
    def _count_depth_frames() -> int:
        if not os.path.isdir(DEPTH_PNG_DIR):
            return 0
        return len([f for f in os.listdir(DEPTH_PNG_DIR) if f.endswith(".png")])

    def make_recon_viz(self) -> bool:
        """포인트 클라우드 + 추출된 seam 원을 다크테마 3D 이미지로 렌더."""
        with self.viz_lock:
            if not os.path.isfile(PLY_PATH):
                return False
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import open3d as o3d
            from welding.seam_extraction import extract_seam

            pcd = o3d.io.read_point_cloud(PLY_PATH)
            pts_all = np.asarray(pcd.points)
            if len(pts_all) == 0:
                return False
            pts = pts_all
            if len(pts) > 30000:
                sel = np.random.default_rng(0).choice(len(pts), 30000, replace=False)
                pts = pts[sel]
            seam = extract_seam(PLY_PATH, verbose=False)
            c = np.asarray(seam["center"], dtype=float)
            r = float(seam["radius"])
            nrm = np.asarray(seam["normal"], dtype=float)
            nrm /= np.linalg.norm(nrm)
            u = np.array([0.0, 1.0, 0.0]) - nrm[1] * nrm
            u /= np.linalg.norm(u)
            v = np.cross(nrm, u)
            th = np.linspace(0, 2 * np.pi, 240)
            circ = c[None, :] + r * (np.cos(th)[:, None] * u[None, :]
                                     + np.sin(th)[:, None] * v[None, :])

            fig = plt.figure(figsize=(6.4, 4.8), facecolor="#FFFFFF")
            ax = fig.add_subplot(111, projection="3d", facecolor="#FFFFFF")
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       s=1.4, c="#3A3A3A", alpha=0.5, linewidths=0)
            ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], c="#3182F7", lw=2.5)
            ax.scatter([c[0]], [c[1]], [c[2]], c="#3182F7", s=36, depthshade=False)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            ctr, half = (lo + hi) / 2, float((hi - lo).max()) / 2 * 1.05
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            ax.set_zlim(ctr[2] - half, ctr[2] + half)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
            ax.view_init(elev=18, azim=-55)
            fig.text(0.04, 0.10, f"SEAM  r = {r * 1000:.2f} mm",
                     color="#3182F7", fontsize=12, fontweight="bold",
                     family="sans-serif")
            fig.text(0.04, 0.04, f"{len(pts_all):,} POINTS",
                     color="#3A3A3A", fontsize=9, family="sans-serif")
            fig.subplots_adjust(left=0, right=1, top=1.05, bottom=0)
            fig.savefig(RECON_VIZ_PATH, dpi=140, facecolor="#FFFFFF")
            plt.close(fig)
            return True

    def set_pipe_distance(self, dist_m: float):
        """로봇 base ↔ 파이프 거리(x)를 라이브 모델에 반영.

        WELD 캐시는 거리별 키로 관리되므로 지우지 않는다 — 같은 거리로
        돌아오면 기존 IK 궤적을 그대로 REPLAY할 수 있다.
        """
        for bid in (self.pipe_bid, self.supp_bid):
            if bid >= 0:
                self.model.body_pos[bid][0] = dist_m
        self.cam.lookat[0] = dist_m * 0.72

    def tip_seam_error(self, seam) -> float:
        R_ee = self.data.xmat[self.ee_bid].reshape(3, 3)
        tip = self.data.site_xpos[self.tcp_sid] + R_ee @ TORCH_TIP_IN_EE
        d = tip - seam["center"]
        nrm = seam["normal"] / np.linalg.norm(seam["normal"])
        ax = float(np.dot(d, nrm))
        rad = float(np.linalg.norm(d - ax * nrm) - seam["radius"])
        return float(np.hypot(ax, rad))

    def _compute_interruptible(self, fn, *args, **kw):
        """블로킹 IK 계산을 워커 스레드에서 실행하며 E-STOP을 폴링한다.

        E-STOP 시 즉시 상태를 갱신하고, model/data 동시 접근을 막기 위해
        계산이 끝날 때까지 join한 뒤 결과를 버리고 None을 반환한다.
        """
        box: dict = {}

        def worker():
            try:
                box["result"] = fn(*args, **kw)
            except Exception as exc:
                box["error"] = exc

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        while th.is_alive():
            if self.estop.is_set():
                self.set_state(status="E-STOP — 진행 중인 IK 계산을 정리하는 중…")
                th.join()
                return None
            time.sleep(0.15)
        if "error" in box:
            raise box["error"]
        return box.get("result")

    # ── 액션 ──────────────────────────────────────────────────────────────
    def run_home(self):
        self.set_state(status="HOME 자세로 복귀 중", action="home", busy=True)
        q0 = np.array([float(self.data.qpos[self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, nm)]])
            for nm in ["q1", "q2", "q3", "q4", "q5", "q6"]])
        q1 = mv.WELD_BIASED_READY
        steps = max(int(2.0 * ANIM_FPS / max(self.speed(), 0.1)), 2)
        for s in range(steps + 1):
            if self.estop.is_set():
                break
            a = s / steps
            a = a * a * (3 - 2 * a)   # smoothstep
            set_arm_qpos(self.model, self.data, mujoco, (1 - a) * q0 + a * q1)
            self.render()
            time.sleep(1.0 / ANIM_FPS)
        self.set_state(status="HOME 위치", action=None, busy=False)

    def run_scan(self):
        self.set_state(status="SCAN: 카메라 궤적 IK 계산 중… (수 분 소요)", action="scan",
                       busy=True, progress={"i": 0, "n": 0})
        self.render()
        # 파이프가 기준 위치(궤적 상수 기준)에서 옮겨져 있으면 스캔 궤적도 함께 이동
        scan_shift = (self.model.body_pos[self.pipe_bid] - mv.WELD_PLY_CAPTURE_PIPE_POS
                      if self.pipe_bid >= 0 else None)

        def ik_prog(i, n):
            self.set_state(progress={"i": i, "n": n},
                           status=f"SCAN: IK 계산 중 — waypoint {i}/{n}")

        result = self._compute_interruptible(
            mv.compute_ik_trajectory, self.model, self.data,
            verbose=False, retries=mv.DEFAULT_IK_RETRIES,
            origin_shift=scan_shift, ik_progress_cb=ik_prog)
        if result is None or self.estop.is_set():
            return
        Q, flags, tvals, angles, cam_pos, cam_rot, cam_poses, tcp_poses, traj_info = result
        self.set_state(status="SCAN: 뎁스 캡처 + 애니메이션 실행 중")
        mv.save_trajectory_csv(META_PATH, Q, flags, angles, cam_poses, tcp_poses, traj_info)

        # 캡처는 model.vis.map.znear/zfar를 D405 뎁스 범위(~0.5m)로 바꾼다.
        # 전역 클리핑이라 그대로 렌더하면 world view가 하늘만 남는다 →
        # UI 렌더 직전에 원래 값으로 되돌리고, 다음 뎁스 프레임 전에 다시 적용.
        znear, zfar = float(self.model.vis.map.znear), float(self.model.vis.map.zfar)
        d405_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "d405_camera")

        def cb(i, n):
            self.set_state(progress={"i": i, "n": n},
                           status=f"SCAN: 캡처 진행 {i} / {n} waypoints")
            self.model.vis.map.znear, self.model.vis.map.zfar = znear, zfar
            self.render(cam=True)
            mv.set_d405_depth_rendering(self.model, d405_id)
            time.sleep(max(1.0 / ANIM_FPS / max(self.speed(), 0.1) - 0.02, 0))
            return not self.estop.is_set()

        try:
            ok = mv.render_inspection_cameras(self.model, self.data, Q, traj_info,
                                              out_dir=os.path.join(ROOT, "inspection_frames"),
                                              progress_cb=cb)
        finally:
            self.model.vis.map.znear, self.model.vis.map.zfar = znear, zfar
        if ok and not self.estop.is_set():
            # 캡처 당시 파이프 위치 기록 → BUILD 후 seam 정합의 기준점
            if self.pipe_bid >= 0:
                with open(os.path.join(ROOT, "inspection_frames", "capture_pipe_pos.json"), "w") as f:
                    json.dump({"pipe_pos": [float(v) for v in self.model.body_pos[self.pipe_bid]]}, f)
            with self.lock:
                self.state["phase_done"]["scan"] = True
                self.state["phase_done"]["build"] = False
                self.state["phase_done"]["weld"] = False
                self.state["n_depth"] = self._count_depth_frames()
                self.state["recon_ready"] = False
            self.set_state(status="✓ SCAN 완료 — BUILD 실행 가능", action=None, busy=False)

    def run_build(self):
        self.set_state(status="BUILD: 3D 형상 복원 중…", action="build", busy=True,
                       progress={"i": 0, "n": 0})
        self.build_proc = subprocess.Popen(RECON_CMD, cwd=ROOT, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
        for line in self.build_proc.stdout:
            line = line.strip()
            if line:
                # "[reconstruct] 41/219 프레임 처리 ..." → 진행 막대
                m = re.search(r"\[reconstruct\]\s*(\d+)\s*/\s*(\d+)", line)
                if m:
                    self.set_state(progress={"i": int(m.group(1)), "n": int(m.group(2))})
                self.set_state(status=f"BUILD: {line[:80]}")
            if self.estop.is_set():
                self.build_proc.terminate()
                return
        self.build_proc.wait()
        if self.build_proc.returncode != 0:
            self.set_state(status="✗ BUILD 실패 — 로그를 확인하세요", action=None, busy=False)
            return
        from welding.seam_extraction import extract_seam
        seam = extract_seam(PLY_PATH, verbose=False)
        self.set_state(seam={"radius_mm": round(seam["radius"] * 1000, 2),
                             "center": [round(float(v), 4) for v in seam["center"]]})
        self.set_state(status="BUILD: seam 시각화 생성 중…")
        try:
            self.make_recon_viz()
        except Exception:
            import traceback; traceback.print_exc()
        with self.lock:
            self.state["phase_done"]["build"] = True
            self.state["phase_done"]["weld"] = False
            self.state["recon_ready"] = True
        self.weld_cache.clear()
        self.set_state(status=f"✓ BUILD 완료 — seam 감지: r={seam['radius']*1000:.2f}mm",
                       action=None, busy=False)

    def run_replay(self):
        """이미 계산된 WELD IK 궤적을 재계산 없이 재생."""
        cached = self.weld_cache.get(self._weld_key())
        if cached is None:
            self.set_state(status="REPLAY 불가 — 이 거리에서 계산된 WELD 궤적이 없습니다 (WELD 먼저 실행)",
                           action=None, busy=False)
            return
        self.set_state(status="REPLAY: 계산된 IK 궤적 재생", action="weld", busy=True)
        self._play_weld(*cached)

    def run_weld(self):
        dist_mm = float(self.params()["pipe_dist_mm"])
        key = self._weld_key()
        cached = self.weld_cache.get(key)
        if cached is None:
            self.set_state(status=f"WELD: 4-segment IK 계산 중… (base↔pipe {dist_mm:.1f}mm, 수 분 소요)",
                           action="weld", busy=True, progress={"i": 0, "n": 0})
            self.render()

            cur_seg = {"name": ""}

            def seg_status(name, i, n):
                cur_seg["name"] = name
                self.set_state(status=f"WELD: IK 계산 중 — segment {i}/{n} ({name})")

            def ik_prog(i, n):
                self.set_state(progress={"i": i, "n": n},
                               status=f"WELD: IK 계산 중 — {cur_seg['name']} · waypoint {i}/{n}")

            result = self._compute_interruptible(
                mv.compute_ik_weld_trajectory, self.model, self.data,
                verbose=False, retries=mv.WELD_DEFAULT_RETRIES,
                status_cb=seg_status, progress_cb=ik_prog)
            if result is None or self.estop.is_set():
                return
            Q, flags, tvals, angles, positions, weld_poses, seam = result
            Q = np.asarray(Q)
            self.weld_cache[key] = (Q, np.asarray(tvals), np.degrees(angles), seam)
            try:
                self._save_weld_cache(key, Q, np.asarray(tvals), np.degrees(angles), seam)
            except Exception as exc:
                print(f"[UI] WELD 궤적 캐시 저장 실패: {exc}")
        else:
            self.set_state(status="WELD: 캐시된 궤적 재생", action="weld", busy=True)
        if self.estop.is_set():
            return
        self._play_weld(*self.weld_cache[key])

    def _play_weld(self, Q, tvals, phi_deg, seam):
        seg_starts = weld_segment_starts(phi_deg)
        n = len(Q)
        # 관절 점프가 큰 구간 = 패스 경계 자세 전환 (토치를 든 상태) → tip error 집계 제외
        dq = np.abs(np.diff(Q, axis=0)).max(axis=1) if n > 1 else np.zeros(1)
        RECONFIG_DQ_RAD = 0.5
        errs: list[float] = []
        t, t_end = 0.0, float(tvals[-1])
        last = time.monotonic()
        while t <= t_end and not self.estop.is_set():
            q = mv.interpolate_q(tvals, Q, t)
            set_arm_qpos(self.model, self.data, mujoco, q)
            wp = int(np.searchsorted(tvals, t, side="right")) - 1
            seg = int(np.searchsorted(seg_starts, wp, side="right")) - 1
            reconfig = wp < len(dq) and dq[min(wp, len(dq) - 1)] > RECONFIG_DQ_RAD
            e = self.tip_seam_error(seam)
            self._cur_tip_err = e            # 모니터 패널 순간값 (재구성 구간 포함)
            if not reconfig:
                errs.append(e)
            tip = {"mean": round(float(np.mean(errs)) * 1000, 3),
                   "max": round(float(np.max(errs)) * 1000, 3)} if errs else \
                  {"mean": None, "max": None}
            status = (f"WELD: 자세 전환 중 (패스 경계) — {wp + 1}/{n}" if reconfig else
                      f"WELD: {SEGMENT_NAMES[max(seg, 0)]} 용접 중 — {wp + 1}/{n}")
            self.set_state(progress={"i": wp + 1, "n": n},
                           segment=max(seg, 0), tip_err=tip, status=status)
            self.render()
            now = time.monotonic()
            dt, last = now - last, now
            t += dt * self.speed()
            time.sleep(max(1.0 / ANIM_FPS - 0.005, 0))
        self._cur_tip_err = None
        if not self.estop.is_set():
            with self.lock:
                self.state["phase_done"]["weld"] = True
            self.set_state(status="✓ 용접 완료", action=None, busy=False, segment=-1)

    # ── 파라미터/유틸 ─────────────────────────────────────────────────────
    def params(self):
        with self.lock:
            return dict(self.state["params"])

    def speed(self) -> float:
        return float(self.params()["speed"])

    def handle_estop(self):
        self.estop.set()
        if self.build_proc is not None and self.build_proc.poll() is None:
            self.build_proc.terminate()

    # ── 메인 루프 (메인 스레드에서 실행) ──────────────────────────────────
    def loop(self):
        set_arm_qpos(self.model, self.data, mujoco, mv.WELD_BIASED_READY)
        self.render()
        actions = {"scan": self.run_scan, "build": self.run_build,
                   "weld": self.run_weld, "replay": self.run_replay,
                   "home": self.run_home}
        while True:
            try:
                cmd = self.cmd_q.get(timeout=1.0 / IDLE_FPS)
            except queue.Empty:
                self.render()
                continue
            if cmd == "estop":
                continue                       # 플래그는 handle_estop에서 이미 처리
            if cmd == "reset":
                with self.lock:
                    self.state["phase_done"] = {"scan": False, "build": False, "weld": False}
                    self.state["seam"] = None
                self.set_state(status="초기화됨", segment=-1,
                               tip_err={"mean": None, "max": None})
                continue
            fn = actions.get(cmd)
            if fn is None:
                continue
            self.estop.clear()
            try:
                fn()
            except Exception as exc:          # UI가 죽지 않도록 방어
                import traceback; traceback.print_exc()
                self.set_state(status=f"✗ 오류: {exc}", action=None, busy=False)
            if self.estop.is_set():
                self.set_state(status="E-STOP — IDLE 상태로 초기화됨", action=None,
                               busy=False, segment=-1, progress={"i": 0, "n": 0})
                self.render()


def make_handler(sim: SimServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # 콘솔 스팸 방지
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(HTML_PATH, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._send(200, sim.get_state().encode())
            elif self.path == "/telemetry":
                with sim.telemetry_lock:
                    samples = list(sim.telemetry)
                    skel = sim.latest_skel
                pipe_x = (float(sim.model.body_pos[sim.pipe_bid][0])
                          if sim.pipe_bid >= 0 else 0.65)
                self._send(200, json.dumps({"samples": samples, "skel": skel,
                                            "pipe_x": round(pipe_x, 4)}).encode())
            elif self.path in ("/stream", "/camstream"):
                name = "world" if self.path == "/stream" else "cam"
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with sim.frame_cond:
                            sim.frame_cond.wait(timeout=2.0)
                            jpeg = sim.jpegs.get(name)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path.startswith("/recon.png"):
                if not os.path.isfile(RECON_VIZ_PATH):
                    try:
                        sim.make_recon_viz()
                    except Exception:
                        pass
                if os.path.isfile(RECON_VIZ_PATH):
                    with open(RECON_VIZ_PATH, "rb") as f:
                        self._send(200, f.read(), "image/png")
                else:
                    self._send(404, b'{"error":"no reconstruction"}')
            elif self.path.startswith("/depthframe/"):
                try:
                    idx = int(self.path.rsplit("/", 1)[1].split("?")[0])
                except ValueError:
                    self._send(400, b'{"error":"bad index"}')
                    return
                p = os.path.join(DEPTH_PNG_DIR, f"frame_{idx:03d}.png")
                if os.path.isfile(p):
                    with open(p, "rb") as f:
                        self._send(200, f.read(), "image/png")
                else:
                    self._send(404, b'{"error":"no frame"}')
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, b'{"error":"bad json"}')
                return
            if self.path == "/action":
                act = payload.get("action")
                if act == "estop":
                    sim.handle_estop()
                    sim.cmd_q.put("estop")
                elif act in ("scan", "build", "weld", "replay", "home", "reset"):
                    if sim.get_state() and json.loads(sim.get_state())["busy"] and act != "reset":
                        self._send(409, b'{"error":"busy"}')
                        return
                    sim.cmd_q.put(act)
                else:
                    self._send(400, b'{"error":"unknown action"}')
                    return
                self._send(200, b'{"ok":true}')
            elif self.path == "/params":
                new_dist = None
                with sim.lock:
                    p = sim.state["params"]
                    if "pipe_dist_mm" in payload:
                        new_dist = float(np.clip(payload["pipe_dist_mm"], 300.0, 900.0))
                        p["pipe_dist_mm"] = round(new_dist, 1)
                    if "speed" in payload:
                        p["speed"] = float(np.clip(payload["speed"], 0.1, 3.0))
                if new_dist is not None:
                    sim.set_pipe_distance(new_dist / 1000.0)
                self._send(200, sim.get_state().encode())
            else:
                self._send(404, b'{"error":"not found"}')

    return Handler


def main():
    ap = argparse.ArgumentParser(description="용접 시뮬레이터 컨트롤 UI")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--scene", default=mv.SCENE_XML)
    args = ap.parse_args()

    if os.path.basename(sys.executable).startswith("mjpython") or "mjpython" in sys.argv[0]:
        print("\n[UI] 경고: mjpython으로 실행하면 오프스크린 렌더가 깨져 화면에 하늘만 나옵니다.")
        print("[UI]       'python weld_ui.py' 로 실행하세요.\n")

    sim = SimServer(os.path.abspath(args.scene))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sim))
    except OSError as exc:
        if exc.errno == 48:  # Address already in use
            print(f"\n[UI] 포트 {args.port}가 이미 사용 중입니다 — 서버가 이미 떠 있을 수 있어요.")
            print(f"[UI]   브라우저에서 http://localhost:{args.port} 를 먼저 열어보고,")
            print(f"[UI]   기존 서버를 내리려면:  pkill -f weld_ui.py")
            print(f"[UI]   다른 포트로 띄우려면:  python weld_ui.py --port {args.port + 1}\n")
            sys.exit(1)
        raise
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"\n  용접 컨트롤 UI:  http://localhost:{args.port}\n")
    try:
        sim.loop()
    except KeyboardInterrupt:
        print("\n[UI] 종료합니다.")


if __name__ == "__main__":
    main()
