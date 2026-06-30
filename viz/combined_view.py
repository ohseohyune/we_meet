"""Single-window combined dashboard + 3D skeleton view."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import numpy as np


def combined_view_available() -> bool:
    try:
        import pyqtgraph  # noqa: F401
        import pyqtgraph.opengl  # noqa: F401
        from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


def _combined_process(
    dash_queue,
    skel_queue,
    ref_path,
    body_names,
    window_seconds: float,
    heatmap: bool,
    fixed_y: bool,
) -> None:
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # ignore Ctrl+C; parent sends SIGTERM

    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Joint Tracking & 3D Skeleton")

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    win.setCentralWidget(splitter)

    # ── Left: dashboard ───────────────────────────────────────────────────────
    dash_widget = QtWidgets.QWidget()
    dash_layout = QtWidgets.QVBoxLayout(dash_widget)

    toolbar = QtWidgets.QHBoxLayout()
    fixed_cb = QtWidgets.QCheckBox("Fixed y")
    fixed_cb.setChecked(bool(fixed_y))
    rmse_label = QtWidgets.QLabel("RMSE: --")
    toolbar.addWidget(fixed_cb)
    toolbar.addStretch(1)
    toolbar.addWidget(rmse_label)
    dash_layout.addLayout(toolbar)

    graphics = pg.GraphicsLayoutWidget()
    dash_layout.addWidget(graphics, 1)

    joint_plots, q_curves, qref_curves, err_curves, err_views = [], [], [], [], []
    for i in range(6):
        plot = graphics.addPlot(row=i // 2, col=i % 2, title=f"q{i + 1}")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "time", units="s")
        plot.setLabel("left", "q", units="rad")
        plot.showAxis("right")
        err_view = pg.ViewBox()
        plot.scene().addItem(err_view)
        plot.getAxis("right").linkToView(err_view)
        err_view.setXLink(plot)

        def _sync(p=plot, v=err_view):
            vb = p.vb if hasattr(p, "vb") else p.getViewBox()
            v.setGeometry(vb.sceneBoundingRect())
            v.linkedViewChanged(vb, v.XAxis)

        plot.vb.sigResized.connect(_sync)
        q_curves.append(plot.plot(pen=pg.mkPen("#2b7bba", width=2)))
        qref_curves.append(plot.plot(pen=pg.mkPen("#111111", width=1.5, style=QtCore.Qt.PenStyle.DashLine)))
        err_curve = pg.PlotCurveItem(pen=pg.mkPen("#d14c32", width=1))
        err_view.addItem(err_curve)
        err_curves.append(err_curve)
        err_views.append(err_view)
        joint_plots.append(plot)

    norm_plot = graphics.addPlot(row=3, col=0, colspan=2, title="||e_q||₂")
    norm_plot.showGrid(x=True, y=True, alpha=0.25)
    norm_plot.setLabel("bottom", "time", units="s")
    norm_plot.setLabel("left", "error", units="rad")
    norm_curve = norm_plot.plot(pen=pg.mkPen("#111111", width=2))
    ex_curve = norm_plot.plot(pen=pg.mkPen("#2f9e44", width=2))

    heat_item = None
    if heatmap:
        hm_plot = graphics.addPlot(row=4, col=0, colspan=2, title="joint error heatmap")
        heat_item = pg.ImageItem()
        hm_plot.addItem(heat_item)

    splitter.addWidget(dash_widget)

    # ── Right: 3D skeleton ────────────────────────────────────────────────────
    view = gl.GLViewWidget()
    view.setCameraPosition(distance=1.5, elevation=20, azimuth=-60)
    view.setBackgroundColor("w")

    # ── MATLAB-style dotted 3D box ────────────────────────────────────────────
    _bx0, _bx1 = -0.10, 1.05
    _by0, _by1 = -0.55, 0.55
    _bz0, _bz1 =  0.00, 1.25
    _gs = 0.10   # grid step
    _ds = 0.016  # dot spacing (~6 dots per 10 cm segment)

    # grid line anchor positions
    _gx = np.round(np.arange(-0.1, 1.01, _gs), 4)
    _gy = np.round(np.arange(-0.5, 0.51, _gs), 4)
    _gz = np.round(np.arange( 0.0, 1.21, _gs), 4)
    # dense dot positions spanning full box extent
    _dx = np.arange(_bx0, _bx1 + _ds / 2, _ds)
    _dy = np.arange(_by0, _by1 + _ds / 2, _ds)
    _dz = np.arange(_bz0, _bz1 + _ds / 2, _ds)

    # build dots for each plane (lines-parallel-to-one-axis at each grid anchor)
    _f1  = np.array([[x, y, _bz0] for y in _gy for x in _dx])   # floor // X
    _f2  = np.array([[x, y, _bz0] for x in _gx for y in _dy])   # floor // Y
    _fw1 = np.array([[x, _by0, z] for z in _gz for x in _dx])   # front wall // X
    _fw2 = np.array([[x, _by0, z] for x in _gx for z in _dz])   # front wall // Z
    _sw1 = np.array([[_bx0, y, z] for z in _gz for y in _dy])   # side wall // Y
    _sw2 = np.array([[_bx0, y, z] for y in _gy for z in _dz])   # side wall // Z
    view.addItem(gl.GLScatterPlotItem(
        pos=np.vstack([_f1, _f2, _fw1, _fw2, _sw1, _sw2]),
        color=(0.25, 0.25, 0.25, 1.0), size=3.0, pxMode=True,
        glOptions='translucent'))

    # box corner edges (solid thin lines)
    _ek = np.array([0.18, 0.18, 0.18, 1.0])
    for _ep in [
        [[_bx0, _by0, _bz0], [_bx1, _by0, _bz0]], [[_bx0, _by0, _bz0], [_bx0, _by1, _bz0]],
        [[_bx0, _by0, _bz0], [_bx0, _by0, _bz1]], [[_bx1, _by0, _bz0], [_bx1, _by1, _bz0]],
        [[_bx0, _by1, _bz0], [_bx1, _by1, _bz0]], [[_bx1, _by0, _bz0], [_bx1, _by0, _bz1]],
        [[_bx0, _by1, _bz0], [_bx0, _by1, _bz1]], [[_bx0, _by0, _bz1], [_bx1, _by0, _bz1]],
        [[_bx0, _by0, _bz1], [_bx0, _by1, _bz1]],
    ]:
        view.addItem(gl.GLLinePlotItem(pos=np.array(_ep, dtype=float), color=_ek, width=1.5, antialias=True))

    # tick marks + numeric labels + axis titles
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
        view.addItem(gl.GLLinePlotItem(pos=np.array([[0, 0, 0], tip], dtype=float), color=np.array(color), width=3, antialias=True))
        if hasattr(gl, "GLTextItem"):
            t = gl.GLTextItem(pos=np.array(tip, dtype=float), text=label, color=(*color[:3], 1.0))
            view.addItem(t)

    ref_path_arr = np.asarray(ref_path, dtype=float)
    past_item = None
    if len(ref_path_arr) >= 2:
        view.addItem(gl.GLLinePlotItem(pos=ref_path_arr, color=(0.0, 0.6, 0.8, 0.30), width=2, antialias=True))
        past_item = gl.GLLinePlotItem(pos=ref_path_arr[:1], color=(0.0, 0.35, 0.9, 0.95), width=4, antialias=True)
        view.addItem(past_item)

    skeleton_item  = gl.GLLinePlotItem(color=(0.10, 0.10, 0.10, 1.0), width=4, antialias=True)
    joint_item     = gl.GLScatterPlotItem(color=(1.00, 0.00, 0.00, 1.0), size=21, pxMode=True)  # 빨간 점
    tcp_item       = gl.GLScatterPlotItem(color=(0.95, 0.15, 0.15, 1.0), size=16, pxMode=True)  # 빨간 점
    target_item    = gl.GLScatterPlotItem(color=(0.05, 0.80, 0.20, 1.0), size=13, pxMode=True)  # 초록 점
    camera_item    = gl.GLScatterPlotItem(color=(1.00, 0.55, 0.00, 1.0), size=16, pxMode=True)  # 주황 점
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

    splitter.addWidget(view)
    splitter.setSizes([780, 600])

    # ── Shared update timer ───────────────────────────────────────────────────
    latest_dash: dict = {}
    latest_skel: dict = {}

    def _drain(q, store):
        while True:
            try:
                store.update(q.get_nowait())
            except queue.Empty:
                break

    def update():
        _drain(dash_queue, latest_dash)
        _drain(skel_queue, latest_skel)

        # dashboard update
        if latest_dash:
            t = latest_dash.get("t", np.array([]))
            if len(t) > 0:
                x = t - t[-1]
                q_data = latest_dash["q"]
                q_ref = latest_dash["q_ref"]
                e_q = latest_dash["e_q"]
                e_q_norm = latest_dash["e_q_norm"]
                e_x_norm = latest_dash.get("e_x_norm", np.full_like(e_q_norm, np.nan))
                for i in range(6):
                    q_curves[i].setData(x, q_data[:, i])
                    qref_curves[i].setData(x, q_ref[:, i])
                    err_curves[i].setData(x, e_q[:, i])
                    joint_plots[i].setXRange(-window_seconds, 0.0, padding=0.0)
                    if fixed_cb.isChecked():
                        joint_plots[i].setYRange(-np.pi, np.pi, padding=0.02)
                        err_views[i].setYRange(-0.25, 0.25, padding=0.02)
                norm_curve.setData(x, e_q_norm)
                if np.isfinite(e_x_norm).any():
                    ex_curve.setData(x, e_x_norm)
                norm_plot.setXRange(-window_seconds, 0.0, padding=0.0)
                rmse = np.sqrt(np.mean(e_q * e_q, axis=0))
                rmse_label.setText("RMSE rad: " + " ".join(f"{v:.4f}" for v in rmse))
                if heat_item is not None:
                    heat_item.setImage(e_q.T, autoLevels=True)

        # skeleton update
        if latest_skel:
            points = np.asarray(latest_skel.get("body_pos", []), dtype=float)
            if points.ndim == 2 and points.shape[1] == 3 and len(points) > 0:
                skeleton_item.setData(pos=points)
                joint_item.setData(pos=points)
                tcp = np.asarray(latest_skel.get("tcp_pos", points[-1]), dtype=float).reshape(1, 3)
                target = np.asarray(latest_skel.get("target_pos", points[-1]), dtype=float).reshape(1, 3)
                tcp_item.setData(pos=tcp)
                target_item.setData(pos=target)
                if past_item is not None and len(ref_path_arr) >= 2:
                    progress = float(latest_skel.get("progress", 0.0))
                    idx = int(np.clip(round(progress * (len(ref_path_arr) - 1)), 0, len(ref_path_arr) - 1))
                    past_item.setData(pos=ref_path_arr[: idx + 1])
                for text, name, pos in zip(text_items, body_names, points):
                    text.setData(pos=pos + np.array([0.0, 0.0, 0.025]), text=str(name))
                # camera
                cam_pos = latest_skel.get("camera_pos")
                cam_dir = latest_skel.get("camera_dir")
                if cam_pos is not None:
                    cam_pos = np.asarray(cam_pos, dtype=float).reshape(1, 3)
                    camera_item.setData(pos=cam_pos)
                    if cam_dir is not None:
                        cam_dir = np.asarray(cam_dir, dtype=float).reshape(1, 3)
                        camera_dir_item.setData(pos=np.vstack([cam_pos, cam_dir]))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(33)

    win.resize(1400, 900)
    win.show()
    app.exec()


class CombinedView:
    """Single-window dashboard + 3D skeleton launched in one subprocess."""

    def __init__(
        self,
        ref_path,
        body_names: tuple[str, ...],
        window_seconds: float = 10.0,
        heatmap: bool = False,
        fixed_y: bool = False,
    ) -> None:
        self.ref_path = np.asarray(ref_path, dtype=float)
        self.body_names = tuple(body_names)
        self.window_seconds = float(window_seconds)
        self.heatmap = bool(heatmap)
        self.fixed_y = bool(fixed_y)
        self._dash_queue: mp.Queue | None = None
        self._skel_queue: mp.Queue | None = None
        self._process: mp.Process | None = None
        self._last_dash = 0.0
        self._last_skel = 0.0

    def start(self) -> bool:
        if not combined_view_available():
            return False
        self._dash_queue = mp.Queue(maxsize=3)
        self._skel_queue = mp.Queue(maxsize=3)
        self._process = mp.Process(
            target=_combined_process,
            args=(
                self._dash_queue,
                self._skel_queue,
                self.ref_path,
                self.body_names,
                self.window_seconds,
                self.heatmap,
                self.fixed_y,
            ),
            daemon=True,
        )
        self._process.start()
        return True

    def _put(self, q: mp.Queue, data: dict, last_attr: str, rate_hz: float) -> None:
        now = time.time()
        if now - getattr(self, last_attr) < 1.0 / max(float(rate_hz), 1.0):
            return
        setattr(self, last_attr, now)
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
        try:
            q.put_nowait(data)
        except queue.Full:
            pass

    def push_dashboard(self, snapshot: dict, rate_hz: float = 30.0) -> None:
        if self._dash_queue is not None:
            self._put(self._dash_queue, snapshot, "_last_dash", rate_hz)

    def push_skeleton(self, state: dict, rate_hz: float = 30.0) -> None:
        if self._skel_queue is not None:
            self._put(self._skel_queue, state, "_last_skel", rate_hz)

    def close(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        # cancel_join_thread() MUST be called before queue GC; otherwise the
        # feeder thread blocks forever on a broken pipe when the subprocess died.
        for q in (self._dash_queue, self._skel_queue):
            if q is not None:
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:
                    pass
        self._process = None
        self._dash_queue = None
        self._skel_queue = None
