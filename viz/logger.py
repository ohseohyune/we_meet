"""Thread-safe ring logger plus full CSV export for tracking diagnostics."""

from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackingSummary:
    samples: int
    duration: float
    joint_rmse: np.ndarray
    joint_max_abs: np.ndarray
    joint_norm_rmse: float
    final_tcp_error: float | None
    mean_tcp_error: float | None
    max_tcp_error: float | None


class RingLogger:
    """Record full tracking logs while exposing a sliding snapshot window."""

    def __init__(
        self,
        window_seconds: float = 10.0,
        nominal_dt: float = 0.002,
        full_history: bool = True,
    ) -> None:
        self.window_seconds = float(window_seconds)
        self.nominal_dt = max(float(nominal_dt), 1.0e-5)
        self.capacity = max(32, int(np.ceil(self.window_seconds / self.nominal_dt)) + 2)
        self.full_history = bool(full_history)
        self._lock = threading.RLock()
        self._cursor = 0
        self._count = 0
        self._t = np.full(self.capacity, np.nan)
        self._q_ref = np.full((self.capacity, 6), np.nan)
        self._q = np.full((self.capacity, 6), np.nan)
        self._p_tcp = np.full((self.capacity, 3), np.nan)
        self._p_ref = np.full((self.capacity, 3), np.nan)
        self._records: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def log(
        self,
        t: float,
        q_ref,
        qpos,
        p_tcp,
        p_ref=None,
    ) -> None:
        """Append one simulation sample.

        q_ref is the existing commanded joint target. If a caller cannot access
        an explicit q_ref, pass data.ctrl[:6] and document that assumption.
        """
        q_ref_arr = np.asarray(q_ref, dtype=float).reshape(6).copy()
        q_arr = np.asarray(qpos, dtype=float).reshape(-1)[:6].copy()
        if q_arr.shape != (6,):
            raise ValueError(f"Expected at least 6 qpos values, got shape {q_arr.shape}")
        p_tcp_arr = np.asarray(p_tcp, dtype=float).reshape(3).copy()
        if p_ref is None:
            p_ref_arr = np.full(3, np.nan)
        else:
            p_ref_arr = np.asarray(p_ref, dtype=float).reshape(3).copy()

        with self._lock:
            idx = self._cursor
            self._t[idx] = float(t)
            self._q_ref[idx] = q_ref_arr
            self._q[idx] = q_arr
            self._p_tcp[idx] = p_tcp_arr
            self._p_ref[idx] = p_ref_arr
            self._cursor = (self._cursor + 1) % self.capacity
            self._count = min(self._count + 1, self.capacity)
            if self.full_history:
                self._records.append((float(t), q_ref_arr, q_arr, p_tcp_arr, p_ref_arr))

    def _ordered_ring_indices(self) -> np.ndarray:
        if self._count == 0:
            return np.array([], dtype=int)
        start = (self._cursor - self._count) % self.capacity
        return (start + np.arange(self._count)) % self.capacity

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return a copy of the sliding-window data sorted by time."""
        with self._lock:
            idx = self._ordered_ring_indices()
            snap = {
                "t": self._t[idx].copy(),
                "q_ref": self._q_ref[idx].copy(),
                "q": self._q[idx].copy(),
                "p_tcp": self._p_tcp[idx].copy(),
                "p_ref": self._p_ref[idx].copy(),
            }
        snap["e_q"] = snap["q_ref"] - snap["q"]
        snap["e_q_norm"] = np.linalg.norm(snap["e_q"], axis=1) if len(snap["e_q"]) else np.array([])
        valid_ref = np.isfinite(snap["p_ref"]).all(axis=1) if len(snap["p_ref"]) else np.array([], dtype=bool)
        e_x = np.full((len(snap["p_tcp"]), 3), np.nan)
        if np.any(valid_ref):
            e_x[valid_ref] = snap["p_ref"][valid_ref] - snap["p_tcp"][valid_ref]
        snap["e_x"] = e_x
        snap["e_x_norm"] = np.linalg.norm(e_x, axis=1)
        return snap

    def records(self) -> dict[str, np.ndarray]:
        """Return a full-history copy for export and summary statistics."""
        with self._lock:
            records = list(self._records)
        if not records:
            return {
                "t": np.array([]),
                "q_ref": np.empty((0, 6)),
                "q": np.empty((0, 6)),
                "p_tcp": np.empty((0, 3)),
                "p_ref": np.empty((0, 3)),
            }
        t = np.array([r[0] for r in records], dtype=float)
        q_ref = np.vstack([r[1] for r in records])
        q = np.vstack([r[2] for r in records])
        p_tcp = np.vstack([r[3] for r in records])
        p_ref = np.vstack([r[4] for r in records])
        return {"t": t, "q_ref": q_ref, "q": q, "p_tcp": p_tcp, "p_ref": p_ref}

    def summary(self) -> TrackingSummary:
        data = self.records()
        t = data["t"]
        q_ref = data["q_ref"]
        q = data["q"]
        p_tcp = data["p_tcp"]
        p_ref = data["p_ref"]
        if len(t) == 0:
            nan6 = np.full(6, np.nan)
            return TrackingSummary(0, 0.0, nan6, nan6, np.nan, None, None, None)

        e_q = q_ref - q
        joint_rmse = np.sqrt(np.mean(e_q * e_q, axis=0))
        joint_max_abs = np.max(np.abs(e_q), axis=0)
        joint_norm_rmse = float(np.sqrt(np.mean(np.sum(e_q * e_q, axis=1))))
        valid_ref = np.isfinite(p_ref).all(axis=1)
        if np.any(valid_ref):
            e_x_norm = np.linalg.norm(p_ref[valid_ref] - p_tcp[valid_ref], axis=1)
            final_tcp_error = float(e_x_norm[-1])
            mean_tcp_error = float(np.mean(e_x_norm))
            max_tcp_error = float(np.max(e_x_norm))
        else:
            final_tcp_error = mean_tcp_error = max_tcp_error = None

        return TrackingSummary(
            samples=len(t),
            duration=float(t[-1] - t[0]) if len(t) > 1 else 0.0,
            joint_rmse=joint_rmse,
            joint_max_abs=joint_max_abs,
            joint_norm_rmse=joint_norm_rmse,
            final_tcp_error=final_tcp_error,
            mean_tcp_error=mean_tcp_error,
            max_tcp_error=max_tcp_error,
        )

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.records()
        t = data["t"]
        q_ref = data["q_ref"]
        q = data["q"]
        p_tcp = data["p_tcp"]
        p_ref = data["p_ref"]
        e_q = q_ref - q
        valid_ref = np.isfinite(p_ref).all(axis=1) if len(p_ref) else np.array([], dtype=bool)
        e_x = np.full_like(p_ref, np.nan)
        if np.any(valid_ref):
            e_x[valid_ref] = p_ref[valid_ref] - p_tcp[valid_ref]

        header = (
            ["t"]
            + [f"q_ref_{i}" for i in range(1, 7)]
            + [f"q_{i}" for i in range(1, 7)]
            + [f"e_q_{i}" for i in range(1, 7)]
            + ["e_q_norm"]
            + ["p_tcp_x", "p_tcp_y", "p_tcp_z"]
            + ["p_ref_x", "p_ref_y", "p_ref_z"]
            + ["e_x", "e_y", "e_z", "e_x_norm"]
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(t)):
                e_q_norm = float(np.linalg.norm(e_q[i]))
                e_x_norm = float(np.linalg.norm(e_x[i])) if np.isfinite(e_x[i]).all() else np.nan
                writer.writerow(
                    [t[i]]
                    + q_ref[i].tolist()
                    + q[i].tolist()
                    + e_q[i].tolist()
                    + [e_q_norm]
                    + p_tcp[i].tolist()
                    + p_ref[i].tolist()
                    + e_x[i].tolist()
                    + [e_x_norm]
                )
