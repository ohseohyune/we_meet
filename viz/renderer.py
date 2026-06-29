"""Passive and offscreen MuJoCo rendering with custom scene overlays."""

from __future__ import annotations

import os
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from .overlay import add_ghost_skeleton, add_reference_overlay


class _FfmpegVideoWriter:
    def __init__(self, path: str, width: int, height: int, fps: float) -> None:
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                f"{float(fps):.6f}",
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def append_data(self, frame) -> None:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.shape[:2] != (self.height, self.width):
            img = Image.fromarray(arr).resize((self.width, self.height))
            arr = np.asarray(img, dtype=np.uint8)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if self.proc.stdin is not None:
            self.proc.stdin.write(np.ascontiguousarray(arr).tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


class Renderer3D:
    """Common API for live passive rendering and offscreen video rendering."""

    def __init__(
        self,
        model,
        data,
        mujoco,
        mode: str,
        ref_path=None,
        out_video: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        fixed_camera_id: int | None = None,
        ghost: bool = False,
        free_camera_lookat=None,
        overlay_enabled: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.mujoco = mujoco
        self.mode = mode
        self.ref_path = np.asarray(ref_path, dtype=float) if ref_path is not None else None
        self.out_video = out_video
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fixed_camera_id = fixed_camera_id
        self.ghost = bool(ghost)
        self.free_camera_lookat = np.asarray(free_camera_lookat, dtype=float) if free_camera_lookat is not None else None
        self.overlay_enabled = bool(overlay_enabled)
        self.viewer = None
        self._viewer_cm = None
        self._renderer = None
        self._writer = None
        self._frame_count = 0
        self._last_sync_wall = 0.0
        self._plot_fallback = False
        self._ghost_data = self.mujoco.MjData(model) if self.ghost else None
        self._ghost_body_names = ("link1", "link2", "link3", "link4", "link5", "link6", "ee")

    def start(self) -> None:
        if self.mode == "live":
            self._viewer_cm = self.mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer = self._viewer_cm.__enter__()
            self.viewer.opt.sitegroup[:] = 1
            if self.fixed_camera_id is not None and self.fixed_camera_id >= 0:
                self.viewer.cam.type = self.mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = int(self.fixed_camera_id)
            else:
                if self.free_camera_lookat is not None:
                    self.viewer.cam.lookat[:] = self.free_camera_lookat
                self.viewer.cam.distance = 1.8
                self.viewer.cam.elevation = -25
                self.viewer.cam.azimuth = 45
            return

        if self.mode != "record":
            raise ValueError(f"Unknown renderer mode: {self.mode}")
        if self.out_video is None:
            raise ValueError("record mode requires out_video")
        os.makedirs(os.path.dirname(self.out_video) or ".", exist_ok=True)
        try:
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(self.out_video, fps=self.fps, codec="libx264", quality=8)
        except Exception as exc:
            print(f"[경고] imageio MP4 writer 초기화 실패({exc}). ffmpeg CLI writer를 사용합니다.")
            self._writer = _FfmpegVideoWriter(self.out_video, self.width, self.height, self.fps)
        if os.environ.get("MUJOCO_GL") == "egl" and not (
            os.path.exists("/dev/dri") or any(os.path.exists(f"/dev/nvidia{i}") for i in range(8))
        ):
            print("[경고] EGL 디바이스가 없어 MuJoCo offscreen renderer 대신 trajectory MP4 fallback을 사용합니다.")
            self._plot_fallback = True
        else:
            try:
                self._renderer = self.mujoco.Renderer(self.model, height=self.height, width=self.width)
            except Exception as exc:
                print(f"[경고] MuJoCo offscreen renderer 초기화 실패({exc}). trajectory MP4 fallback을 사용합니다.")
                self._renderer = None
                self._plot_fallback = True

    def is_running(self) -> bool:
        if self.mode == "live":
            return bool(self.viewer is not None and self.viewer.is_running())
        return True

    def _progress_index(self, progress: float | None) -> int | None:
        if progress is None or self.ref_path is None or len(self.ref_path) == 0:
            return None
        return int(np.clip(round(float(progress) * (len(self.ref_path) - 1)), 0, len(self.ref_path) - 1))

    def _update_overlay(self, scene, tcp_pos=None, target_pos=None, q_ref=None, progress: float | None = None) -> None:
        progress_index = self._progress_index(progress)
        add_reference_overlay(
            scene,
            self.mujoco,
            ref_path=self.ref_path,
            progress_index=progress_index,
            tcp_pos=tcp_pos,
            target_pos=target_pos,
        )
        if self.ghost and q_ref is not None and self._ghost_data is not None:
            self._ghost_data.qpos[:6] = np.asarray(q_ref, dtype=float).reshape(6)
            self.mujoco.mj_forward(self.model, self._ghost_data)
            add_ghost_skeleton(scene, self.mujoco, self.model, self._ghost_data, self._ghost_body_names)

    def sync(self, tcp_pos=None, target_pos=None, q_ref=None, progress: float | None = None) -> None:
        if self.mode != "live" or self.viewer is None:
            return
        now = time.time()
        if now - self._last_sync_wall < 1.0 / 60.0:
            return
        self._last_sync_wall = now
        if self.overlay_enabled:
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0
                self._update_overlay(self.viewer.user_scn, tcp_pos=tcp_pos, target_pos=target_pos, q_ref=q_ref, progress=progress)
        self.viewer.sync()

    def render_frame(self, tcp_pos=None, target_pos=None, q_ref=None, progress: float | None = None) -> None:
        if self.mode != "record" or self._writer is None:
            return
        if self._plot_fallback:
            self._writer.append_data(self._render_plot_fallback(tcp_pos=tcp_pos, target_pos=target_pos, progress=progress))
            self._frame_count += 1
            return
        if self._renderer is None:
            return
        camera = self.fixed_camera_id if self.fixed_camera_id is not None and self.fixed_camera_id >= 0 else None
        if camera is None:
            self._renderer.update_scene(self.data)
        else:
            self._renderer.update_scene(self.data, camera=camera)
        self._update_overlay(self._renderer.scene, tcp_pos=tcp_pos, target_pos=target_pos, q_ref=q_ref, progress=progress)
        frame = self._renderer.render()
        self._writer.append_data(frame)
        self._frame_count += 1

    def _project_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.ref_path is not None and len(self.ref_path):
            base = self.ref_path
        else:
            base = points
        valid = np.isfinite(base).all(axis=1)
        if not np.any(valid):
            center = np.zeros(3)
            span = 1.0
        else:
            cloud = base[valid]
            center = 0.5 * (np.min(cloud, axis=0) + np.max(cloud, axis=0))
            span = float(np.max(np.ptp(cloud, axis=0)))
            span = max(span, 1.0e-3)
        rel = points - center
        x2 = 0.90 * rel[:, 0] - 0.42 * rel[:, 1]
        y2 = -0.25 * rel[:, 0] - 0.45 * rel[:, 1] + 0.95 * rel[:, 2]
        scale = 0.78 * min(self.width, self.height) / span
        px = self.width * 0.5 + scale * x2
        py = self.height * 0.52 - scale * y2
        return np.column_stack([px, py])

    def _draw_polyline(self, draw, points, fill, width: int) -> None:
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            return
        pts = self._project_points(points)
        pairs = [tuple(map(float, p)) for p in pts]
        draw.line(pairs, fill=fill, width=width, joint="curve")

    def _render_plot_fallback(self, tcp_pos=None, target_pos=None, progress: float | None = None):
        img = Image.new("RGB", (self.width, self.height), (248, 249, 250))
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle((0, 0, self.width, self.height), fill=(248, 249, 250, 255))
        draw.text((18, 14), "MuJoCo offscreen unavailable: trajectory fallback", fill=(33, 37, 41, 255))
        if self.ref_path is not None and len(self.ref_path) >= 2:
            idx = self._progress_index(progress) or 0
            self._draw_polyline(draw, self.ref_path[idx:], fill=(0, 170, 210, 65), width=2)
            self._draw_polyline(draw, self.ref_path[: idx + 1], fill=(0, 150, 210, 230), width=4)
        for pos, color in ((target_pos, (20, 210, 70, 255)), (tcp_pos, (235, 45, 35, 255))):
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=float)
            if not np.isfinite(pos).all():
                continue
            x, y = self._project_points(pos.reshape(1, 3))[0]
            r = 6
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        return np.asarray(img)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer_cm is not None:
            self._viewer_cm.__exit__(None, None, None)
            self._viewer_cm = None
            self.viewer = None
