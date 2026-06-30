"""Separate pyqtgraph/OpenGL 3D skeleton view for q_ref tracking."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import numpy as np


def skeleton3d_available() -> bool:
    try:
        import pyqtgraph  # noqa: F401
        import pyqtgraph.opengl  # noqa: F401
        from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


def _axis_item(gl, start, end, color):
    item = gl.GLLinePlotItem(
        pos=np.asarray([start, end], dtype=float),
        color=np.asarray(color, dtype=float),
        width=3,
        antialias=True,
    )
    return item


def _skeleton_process(state_queue, ref_path, body_names, window_title: str) -> None:
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # ignore Ctrl+C; parent sends SIGTERM

    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = gl.GLViewWidget()
    view.setWindowTitle(window_title)
    view.setCameraPosition(distance=1.5, elevation=20, azimuth=-60)
    view.setBackgroundColor("w")

    # ── MATLAB-style dotted 3D box ────────────────────────────────────────────
    _bx0, _bx1 = -0.10, 1.05
    _by0, _by1 = -0.55, 0.55
    _bz0, _bz1 =  0.00, 1.25
    _gs = 0.10
    _ds = 0.016

    _gx = np.round(np.arange(-0.1, 1.01, _gs), 4)
    _gy = np.round(np.arange(-0.5, 0.51, _gs), 4)
    _gz = np.round(np.arange( 0.0, 1.21, _gs), 4)
    _dx = np.arange(_bx0, _bx1 + _ds / 2, _ds)
    _dy = np.arange(_by0, _by1 + _ds / 2, _ds)
    _dz = np.arange(_bz0, _bz1 + _ds / 2, _ds)

    _f1  = np.array([[x, y, _bz0] for y in _gy for x in _dx])
    _f2  = np.array([[x, y, _bz0] for x in _gx for y in _dy])
    _fw1 = np.array([[x, _by0, z] for z in _gz for x in _dx])
    _fw2 = np.array([[x, _by0, z] for x in _gx for z in _dz])
    _sw1 = np.array([[_bx0, y, z] for z in _gz for y in _dy])
    _sw2 = np.array([[_bx0, y, z] for y in _gy for z in _dz])
    view.addItem(gl.GLScatterPlotItem(
        pos=np.vstack([_f1, _f2, _fw1, _fw2, _sw1, _sw2]),
        color=(0.25, 0.25, 0.25, 1.0), size=3.0, pxMode=True,
        glOptions='translucent'))

    _ek = np.array([0.18, 0.18, 0.18, 1.0])
    for _ep in [
        [[_bx0, _by0, _bz0], [_bx1, _by0, _bz0]], [[_bx0, _by0, _bz0], [_bx0, _by1, _bz0]],
        [[_bx0, _by0, _bz0], [_bx0, _by0, _bz1]], [[_bx1, _by0, _bz0], [_bx1, _by1, _bz0]],
        [[_bx0, _by1, _bz0], [_bx1, _by1, _bz0]], [[_bx1, _by0, _bz0], [_bx1, _by0, _bz1]],
        [[_bx0, _by1, _bz0], [_bx0, _by1, _bz1]], [[_bx0, _by0, _bz1], [_bx1, _by0, _bz1]],
        [[_bx0, _by0, _bz1], [_bx0, _by1, _bz1]],
    ]:
        view.addItem(gl.GLLinePlotItem(pos=np.array(_ep, dtype=float), color=_ek, width=1.5, antialias=True))

    _xt = [round(v, 2) for v in np.arange(0.0,  _bx1 + 0.001, 0.20) if _bx0 <= round(v, 4) <= _bx1]
    _yt = [round(v, 2) for v in np.arange(-0.4, _by1 + 0.001, 0.20) if _by0 <= round(v, 4) <= _by1]
    _zt = [round(v, 2) for v in np.arange(0.0,  _bz1 + 0.001, 0.25) if _bz0 <= round(v, 4) <= _bz1]
    for _xv in _xt:
        view.addItem(gl.GLLinePlotItem(pos=np.array([[_xv, _by0, _bz0], [_xv, _by0 - 0.022, _bz0]], dtype=float), color=_ek, width=1.1, antialias=True))
    for _yv in _yt:
        view.addItem(gl.GLLinePlotItem(pos=np.array([[_bx0, _yv, _bz0], [_bx0 - 0.022, _yv, _bz0]], dtype=float), color=_ek, width=1.1, antialias=True))
    for _zv in _zt:
        view.addItem(gl.GLLinePlotItem(pos=np.array([[_bx0, _by0, _zv], [_bx0, _by0 - 0.022, _zv]], dtype=float), color=_ek, width=1.1, antialias=True))
    if hasattr(gl, "GLTextItem"):
        _tc = (0.15, 0.15, 0.15, 1.0)
        _ac = (0.05, 0.05, 0.05, 1.0)
        for _xv in _xt:
            view.addItem(gl.GLTextItem(pos=np.array([_xv, _by0 - 0.07, _bz0], dtype=float), text=f"{_xv:.1f}", color=_tc))
        view.addItem(gl.GLTextItem(pos=np.array([(_bx0+_bx1)/2, _by0 - 0.17, _bz0], dtype=float), text="X [m]", color=_ac))
        for _yv in _yt:
            view.addItem(gl.GLTextItem(pos=np.array([_bx0 - 0.08, _yv, _bz0], dtype=float), text=f"{_yv:.1f}", color=_tc))
        view.addItem(gl.GLTextItem(pos=np.array([_bx0 - 0.17, (_by0+_by1)/2, _bz0], dtype=float), text="Y [m]", color=_ac))
        for _zv in _zt:
            view.addItem(gl.GLTextItem(pos=np.array([_bx0 - 0.07, _by0 - 0.04, _zv], dtype=float), text=f"{_zv:.2f}", color=_tc))
        view.addItem(gl.GLTextItem(pos=np.array([_bx0 - 0.13, _by0 - 0.10, (_bz0+_bz1)/2], dtype=float), text="Z [m]", color=_ac))

    _axis_defs = [
        ([0.4, 0, 0], [1.0, 0.0, 0.0, 1.0], "X"),
        ([0, 0.4, 0], [0.0, 0.65, 0.0, 1.0], "Y"),
        ([0, 0, 0.4], [0.0, 0.15, 1.0, 1.0], "Z"),
    ]
    for tip, color, label in _axis_defs:
        view.addItem(_axis_item(gl, [0, 0, 0], tip, color))
        if hasattr(gl, "GLTextItem"):
            t = gl.GLTextItem(pos=np.array(tip, dtype=float), text=label, color=(*color[:3], 1.0))
            view.addItem(t)

    ref_path = np.asarray(ref_path, dtype=float)
    if len(ref_path) >= 2:
        ref_item = gl.GLLinePlotItem(pos=ref_path, color=(0.0, 0.6, 0.8, 0.30), width=2, antialias=True)
        view.addItem(ref_item)
        past_item = gl.GLLinePlotItem(pos=ref_path[:1], color=(0.0, 0.35, 0.9, 0.95), width=4, antialias=True)
        view.addItem(past_item)
    else:
        past_item = None

    skeleton_item   = gl.GLLinePlotItem(color=(0.10, 0.10, 0.10, 1.0), width=4, antialias=True)
    joint_item      = gl.GLScatterPlotItem(color=(1.00, 0.00, 0.00, 1.0), size=21, pxMode=True)
    tcp_item        = gl.GLScatterPlotItem(color=(0.95, 0.15, 0.15, 1.0), size=16, pxMode=True)
    target_item     = gl.GLScatterPlotItem(color=(0.05, 0.80, 0.20, 1.0), size=13, pxMode=True)
    camera_item     = gl.GLScatterPlotItem(color=(1.00, 0.55, 0.00, 1.0), size=16, pxMode=True)
    _dummy2 = np.zeros((2, 3), dtype=float)
    camera_dir_item = gl.GLLinePlotItem(pos=_dummy2, color=(1.0, 0.55, 0.0, 0.85), width=2, antialias=True)
    for item in (skeleton_item, joint_item, tcp_item, target_item, camera_item, camera_dir_item):
        view.addItem(item)

    text_items = []
    if hasattr(gl, "GLTextItem"):
        for name in body_names:
            text = gl.GLTextItem(text=str(name), color=(0.0, 0.0, 0.0, 0.8))
            view.addItem(text)
            text_items.append(text)

    latest = {}

    def drain_queue():
        nonlocal latest
        while True:
            try:
                latest = state_queue.get_nowait()
            except queue.Empty:
                break

    def update():
        drain_queue()
        if not latest:
            return
        points = np.asarray(latest.get("body_pos", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            return
        skeleton_item.setData(pos=points)
        joint_item.setData(pos=points)
        tcp = np.asarray(latest.get("tcp_pos", points[-1]), dtype=float).reshape(1, 3)
        target = np.asarray(latest.get("target_pos", points[-1]), dtype=float).reshape(1, 3)
        tcp_item.setData(pos=tcp)
        target_item.setData(pos=target)
        if past_item is not None and len(ref_path) >= 2:
            progress = float(latest.get("progress", 0.0))
            idx = int(np.clip(round(progress * (len(ref_path) - 1)), 0, len(ref_path) - 1))
            past_item.setData(pos=ref_path[: idx + 1])
        for text, name, pos in zip(text_items, body_names, points):
            text.setData(pos=pos + np.array([0.0, 0.0, 0.025]), text=str(name))
        cam_pos = latest.get("camera_pos")
        cam_dir = latest.get("camera_dir")
        if cam_pos is not None:
            cam_pos = np.asarray(cam_pos, dtype=float).reshape(1, 3)
            camera_item.setData(pos=cam_pos)
            if cam_dir is not None:
                cam_dir = np.asarray(cam_dir, dtype=float).reshape(1, 3)
                camera_dir_item.setData(pos=np.vstack([cam_pos, cam_dir]))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(33)
    view.resize(820, 720)
    view.show()
    app.exec()


class Skeleton3D:
    """Own a separate 3D skeleton process so MuJoCo viewer remains untouched."""

    def __init__(self, ref_path, body_names: tuple[str, ...]) -> None:
        self.ref_path = np.asarray(ref_path, dtype=float)
        self.body_names = tuple(body_names)
        self._queue = None
        self._process = None
        self._last_push = 0.0

    def start(self) -> bool:
        if not skeleton3d_available():
            return False
        self._queue = mp.Queue(maxsize=3)
        self._process = mp.Process(
            target=_skeleton_process,
            args=(self._queue, self.ref_path, self.body_names, "3D Skeleton + Reference Trajectory"),
            daemon=True,
        )
        self._process.start()
        return True

    def push(self, state: dict[str, np.ndarray], rate_hz: float = 30.0) -> None:
        if self._queue is None:
            return
        now = time.time()
        if now - self._last_push < 1.0 / max(float(rate_hz), 1.0):
            return
        self._last_push = now
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(state)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        if self._queue is not None:
            try:
                self._queue.cancel_join_thread()
                self._queue.close()
            except Exception:
                pass
        self._process = None
        self._queue = None
