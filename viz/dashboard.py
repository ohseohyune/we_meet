"""pyqtgraph live dashboard and matplotlib record summaries."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time

import numpy as np


def dashboard_available() -> bool:
    try:
        import pyqtgraph  # noqa: F401
        from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: F401
    except Exception:
        return False
    return True


def _dashboard_process(snapshot_queue, window_seconds: float, heatmap: bool, fixed_y: bool) -> None:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    win.setWindowTitle("Joint Tracking Dashboard")
    root = QtWidgets.QVBoxLayout(win)
    toolbar = QtWidgets.QHBoxLayout()
    fixed_cb = QtWidgets.QCheckBox("Fixed y")
    fixed_cb.setChecked(bool(fixed_y))
    rmse_label = QtWidgets.QLabel("RMSE: --")
    toolbar.addWidget(fixed_cb)
    toolbar.addStretch(1)
    toolbar.addWidget(rmse_label)
    root.addLayout(toolbar)

    graphics = pg.GraphicsLayoutWidget()
    root.addWidget(graphics, 1)
    joint_plots = []
    q_curves = []
    qref_curves = []
    err_curves = []
    err_views = []
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

        def update_views(*args, p=plot, v=err_view):
            view_box = p.vb if hasattr(p, "vb") else p.getViewBox()
            v.setGeometry(view_box.sceneBoundingRect())
            v.linkedViewChanged(view_box, v.XAxis)

        plot.vb.sigResized.connect(update_views)
        q_curves.append(plot.plot(pen=pg.mkPen("#2b7bba", width=2)))
        qref_curves.append(plot.plot(pen=pg.mkPen("#111111", width=1.5, style=QtCore.Qt.PenStyle.DashLine)))
        err_curve = pg.PlotCurveItem(pen=pg.mkPen("#d14c32", width=1))
        err_view.addItem(err_curve)
        err_curves.append(err_curve)
        err_views.append(err_view)
        joint_plots.append(plot)

    norm_plot = graphics.addPlot(row=3, col=0, colspan=2, title="||e_q||2")
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

    latest = {}

    def drain_queue():
        nonlocal latest
        while True:
            try:
                latest = snapshot_queue.get_nowait()
            except queue.Empty:
                break

    def update():
        drain_queue()
        if not latest:
            return
        t = latest.get("t", np.array([]))
        if len(t) == 0:
            return
        x = t - t[-1]
        q = latest["q"]
        q_ref = latest["q_ref"]
        e_q = latest["e_q"]
        e_q_norm = latest["e_q_norm"]
        e_x_norm = latest.get("e_x_norm", np.full_like(e_q_norm, np.nan))
        for i in range(6):
            q_curves[i].setData(x, q[:, i])
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

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(33)
    win.resize(1180, 900)
    win.show()
    app.exec()


class Dashboard:
    """Launch a pyqtgraph dashboard in a separate process."""

    def __init__(self, window_seconds: float = 10.0, heatmap: bool = False, fixed_y: bool = False) -> None:
        self.window_seconds = float(window_seconds)
        self.heatmap = bool(heatmap)
        self.fixed_y = bool(fixed_y)
        self._queue = None
        self._process = None
        self._last_push = 0.0

    def start(self) -> bool:
        if not dashboard_available():
            return False
        self._queue = mp.Queue(maxsize=3)
        self._process = mp.Process(
            target=_dashboard_process,
            args=(self._queue, self.window_seconds, self.heatmap, self.fixed_y),
            daemon=True,
        )
        self._process.start()
        return True

    def push(self, snapshot: dict[str, np.ndarray], rate_hz: float = 30.0) -> None:
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
            self._queue.put_nowait(snapshot)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._process = None
        self._queue = None


def save_summary_png(records: dict[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = records["t"]
    q_ref = records["q_ref"]
    q = records["q"]
    p_tcp = records["p_tcp"]
    p_ref = records["p_ref"]
    e_q = q_ref - q
    valid_ref = np.isfinite(p_ref).all(axis=1) if len(p_ref) else np.array([], dtype=bool)
    e_x_norm = np.full(len(t), np.nan)
    if np.any(valid_ref):
        e_x_norm[valid_ref] = np.linalg.norm(p_ref[valid_ref] - p_tcp[valid_ref], axis=1)

    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
    axes = axes.ravel()
    for i in range(6):
        axes[i].plot(t, q[:, i], label=f"q{i + 1}")
        axes[i].plot(t, q_ref[:, i], "--", label=f"q_ref{i + 1}")
        axes[i].set_ylabel("rad")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=7)
    axes[6].plot(t, np.linalg.norm(e_q, axis=1), label="||e_q||2")
    if np.isfinite(e_x_norm).any():
        axes[6].plot(t, e_x_norm, label="||e_x||2")
    axes[6].set_ylabel("error")
    axes[6].grid(True, alpha=0.3)
    axes[6].legend(fontsize=8)
    rmse = np.sqrt(np.mean(e_q * e_q, axis=0)) if len(e_q) else np.full(6, np.nan)
    axes[7].bar(np.arange(1, 7), rmse)
    axes[7].set_xlabel("joint")
    axes[7].set_ylabel("RMSE [rad]")
    axes[7].grid(True, axis="y", alpha=0.3)
    for ax in axes[:7]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
