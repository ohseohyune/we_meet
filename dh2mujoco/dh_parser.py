"""DH table parsing and forward kinematics across all supported modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import Config, DHColumnOrder
from .kinematics import (
    modified_dh_transform,
    standard_dh_transform,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DHRow:
    """One row of DH parameters after column remapping."""

    a: float      # link length  (a_{i-1} for Modified DH, a_i for Standard DH)
    alpha: float  # link twist   (alpha_{i-1} / alpha_i)
    d: float      # joint offset d_i
    theta: float  # joint angle or offset theta_i


@dataclass
class DHTable:
    """Full DH table for a serial manipulator.

    Attributes
    ----------
    rows :    All DH rows (joints + fixed frames).
    n_joints: How many of the first rows correspond to revolute joints.
              The remaining rows are fixed frames (e.g., an EE transform).
    """

    rows: List[DHRow]
    n_joints: int

    @property
    def joint_rows(self) -> List[DHRow]:
        return self.rows[: self.n_joints]

    @property
    def ee_rows(self) -> List[DHRow]:
        return self.rows[self.n_joints :]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_dh_array(
    raw: np.ndarray,
    column_order: DHColumnOrder,
    n_joints: int,
) -> DHTable:
    """Parse a raw (N, 4) DH array into a :class:`DHTable`.

    Parameters
    ----------
    raw :
        Shape (N, 4) or flat (4*N,) array of DH values.
    column_order :
        Mapping from semantic name to column index.
    n_joints :
        Number of joint rows (remaining rows are fixed EE frames).
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 4)

    rows = [
        DHRow(
            a=float(row[column_order.a]),
            alpha=float(row[column_order.alpha]),
            d=float(row[column_order.d]),
            theta=float(row[column_order.theta]),
        )
        for row in raw
    ]
    return DHTable(rows=rows, n_joints=n_joints)


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def _single_transform(
    row: DHRow,
    joint_angle: float,
    mode: int,
    joint_offset_override: float | None = None,
) -> np.ndarray:
    """Compute the 4×4 transform for one DH row given a joint angle.

    Parameters
    ----------
    row :
        DH parameters for this link.
    joint_angle :
        Current joint variable (q_i).  For fixed frames, pass 0.0.
    mode :
        DH convention mode (1–4).
    joint_offset_override :
        When mode == 4, the offset comes from an external list rather than
        the table's theta column.
    """
    if mode == 1:
        theta = row.theta + joint_angle
        return modified_dh_transform(row.a, row.alpha, row.d, theta)

    elif mode == 2:
        # theta in the table is treated as the *current* angle; zero it out
        theta = joint_angle
        return modified_dh_transform(row.a, row.alpha, row.d, theta)

    elif mode == 3:
        theta = row.theta + joint_angle
        return standard_dh_transform(row.a, row.alpha, row.d, theta)

    elif mode == 4:
        offset = joint_offset_override if joint_offset_override is not None else 0.0
        theta = offset + joint_angle
        return modified_dh_transform(row.a, row.alpha, row.d, theta)

    else:
        raise ValueError(f"Unknown DH mode: {mode}")


def compute_dh_fk(
    table: DHTable,
    q: np.ndarray,
    config: Config,
    return_chain: bool = False,
) -> np.ndarray | List[np.ndarray]:
    """Forward kinematics from DH parameters.

    Parameters
    ----------
    table :
        Parsed DH table.
    q :
        Joint angles (n_joints,).
    config :
        Active configuration.
    return_chain :
        If True, return a list of 4×4 transforms T_0, T_1, … T_ee instead
        of just the final EE transform.

    Returns
    -------
    np.ndarray or list of np.ndarray
        End-effector transform, or full chain when *return_chain* is True.
    """
    q = np.asarray(q, dtype=float)
    if q.shape[0] != table.n_joints:
        raise ValueError(
            f"Expected {table.n_joints} joint angles, got {q.shape[0]}"
        )

    T = np.eye(4, dtype=float)
    chain: List[np.ndarray] = [T.copy()]

    # Revolute joints
    for i, (row, qi) in enumerate(zip(table.joint_rows, q)):
        offset_override = (
            config.joint_offsets[i]
            if config.MODE == 4 and i < len(config.joint_offsets)
            else None
        )
        T = T @ _single_transform(row, qi, config.MODE, offset_override)
        chain.append(T.copy())

    # Fixed EE frames (no joint variable, angle = 0)
    for row in table.ee_rows:
        # EE rows always use Modified DH transform regardless of mode,
        # because they encode a fixed geometric offset (no joint DOF).
        T = T @ modified_dh_transform(row.a, row.alpha, row.d, row.theta)
        chain.append(T.copy())

    return chain if return_chain else T


def theta_offsets_from_table(table: DHTable) -> List[float]:
    """Extract joint offset angles from the DH table (column theta of joint rows)."""
    return [row.theta for row in table.joint_rows]
