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
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    view = gl.GLViewWidget()
    view.setWindowTitle(window_title)
    view.setCameraPosition(distance=1.5, elevation=20, azimuth=-60)
    view.setBackgroundColor("w")

    grid = gl.GLGridItem()
    grid.setSize(x=1.0, y=1.0, z=1.0)
    grid.setSpacing(x=0.1, y=0.1, z=0.1)
    grid.setColor((0.65, 0.65, 0.65, 0.3))
    view.addItem(grid)

    plane_verts = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=float,
    )
    plane_faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    plane_mesh = gl.MeshData(vertexes=plane_verts, faces=plane_faces)
    plane_item = gl.GLMeshItem(meshdata=plane_mesh, smooth=False, color=(0.8, 0.95, 0.8, 0.22), shader="shaded", glOptions="opaque")
    view.addItem(plane_item)

    view.addItem(_axis_item(gl, [0, 0, 0], [0.4, 0, 0], [1.0, 0.0, 0.0, 1.0]))
    view.addItem(_axis_item(gl, [0, 0, 0], [0, 0.4, 0], [0.0, 0.65, 0.0, 1.0]))
    view.addItem(_axis_item(gl, [0, 0, 0], [0, 0, 0.4], [0.0, 0.15, 1.0, 1.0]))

    ref_path = np.asarray(ref_path, dtype=float)
    if len(ref_path) >= 2:
        ref_item = gl.GLLinePlotItem(pos=ref_path, color=(0.0, 0.6, 0.8, 0.30), width=2, antialias=True)
        view.addItem(ref_item)
        past_item = gl.GLLinePlotItem(pos=ref_path[:1], color=(0.0, 0.35, 0.9, 0.95), width=4, antialias=True)
        view.addItem(past_item)
    else:
        past_item = None

    skeleton_item = gl.GLLinePlotItem(color=(0.05, 0.05, 0.05, 1.0), width=5, antialias=True)
    joint_item = gl.GLScatterPlotItem(color=(0.15, 0.15, 0.15, 1.0), size=10, pxMode=True)
    tcp_item = gl.GLScatterPlotItem(color=(1.0, 0.2, 0.2, 1.0), size=14, pxMode=True)
    target_item = gl.GLScatterPlotItem(color=(0.0, 0.8, 0.2, 1.0), size=12, pxMode=True)
    view.addItem(skeleton_item)
    view.addItem(joint_item)
    view.addItem(tcp_item)
    view.addItem(target_item)

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
        self._process = None
        self._queue = None
