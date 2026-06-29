"""FK verification: DH chain vs. simulated MuJoCo body hierarchy.

Strategy
--------
1. Compute FK directly from DH parameters using :func:`compute_dh_fk`.
2. Compute FK by replaying the body chain that :class:`MJCFWriter` produced,
   via :func:`simulated_mjcf_fk`.  This mirrors exactly what MuJoCo would
   compute internally.
3. Compare at every frame and report position / orientation errors.

Targets:  position error < 1e-8 m,  orientation error < 1e-8 rad.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .config import Config
from .dh_parser import DHTable, compute_dh_fk
from .kinematics import extract_position, extract_rotation
from .mjcf_writer import BodyFrame, simulated_mjcf_fk
from .quaternion import rotation_matrix_to_quat, quat_angle_between

_POS_TOL = 1e-8   # metres
_ORI_TOL = 1e-8   # radians


def _orientation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle [rad] between two rotation matrices."""
    q1 = rotation_matrix_to_quat(R1)
    q2 = rotation_matrix_to_quat(R2)
    return quat_angle_between(q1, q2)


def _pos_error(T1: np.ndarray, T2: np.ndarray) -> float:
    return float(np.linalg.norm(T1[:3, 3] - T2[:3, 3]))


def verify(
    table: DHTable,
    config: Config,
    body_chain: List[BodyFrame],
    q: Optional[np.ndarray] = None,
    label: str = "",
) -> Tuple[float, float]:
    """Compare DH FK with simulated MJCF FK.

    Parameters
    ----------
    table :      Parsed DH table.
    config :     Active config.
    body_chain : Body frames returned by :class:`MJCFWriter`.
    q :          Joint angles.  Defaults to all-zeros.
    label :      Title string for the printout.

    Returns
    -------
    (max_pos_error_m, max_ori_error_rad) over all frames.
    """
    if q is None:
        q = np.zeros(table.n_joints)

    # --- DH FK chain (one transform per row including EE rows) ----------
    dh_chain: List[np.ndarray] = compute_dh_fk(
        table, q, config, return_chain=True
    )  # type: ignore[arg-type]

    # --- Simulated MJCF FK chain ----------------------------------------
    # body_chain has 2 entries per DH row (pre + post bodies), so we take
    # every second element starting from index 1 to get the "post" body
    # transforms that correspond to each DH frame.
    mjcf_chain: List[np.ndarray] = simulated_mjcf_fk(body_chain, q)

    # mjcf_chain[0] = identity (base)
    # mjcf_chain[1] = after pre_1  (NOT a DH frame)
    # mjcf_chain[2] = after post_1 = DH frame T_1
    # mjcf_chain[3] = after pre_2
    # mjcf_chain[4] = after post_2 = DH frame T_2  … etc.
    # dh_chain[0]  = identity,  dh_chain[1] = T_1,  dh_chain[2] = T_2 …

    total_dh_rows = table.n_joints + len(table.ee_rows)
    # We expect 2 body entries per DH row (pre + post).
    expected_chain_len = 2 * total_dh_rows + 1  # +1 for the root identity

    # Indices in mjcf_chain that correspond to DH frames T_1 … T_N:
    # frame k (1-indexed) is at mjcf_chain[2*k]
    dh_indices = [2 * k for k in range(1, total_dh_rows + 1)]

    print(f"\n{'='*62}")
    print(f"  FK Verification  {label}")
    print(f"  q = {np.round(q, 4).tolist()}")
    print(f"{'='*62}")
    print(f"  {'Frame':<12}  {'pos_err [m]':<20}  {'ori_err [rad]':<18}  Status")
    print(f"  {'-'*60}")

    max_pos_err = 0.0
    max_ori_err = 0.0
    first_divergence: Optional[int] = None

    for k in range(1, total_dh_rows + 1):
        dh_T = dh_chain[k]
        mjcf_idx = dh_indices[k - 1]

        if mjcf_idx >= len(mjcf_chain):
            print(f"  Frame {k:<7}  [chain length mismatch – mjcf_chain too short]")
            continue

        mjcf_T = mjcf_chain[mjcf_idx]
        pe = _pos_error(dh_T, mjcf_T)
        oe = _orientation_error(extract_rotation(dh_T), extract_rotation(mjcf_T))

        max_pos_err = max(max_pos_err, pe)
        max_ori_err = max(max_ori_err, oe)

        ok_p = pe < _POS_TOL
        ok_o = oe < _ORI_TOL
        status = "OK" if (ok_p and ok_o) else "FAIL"

        if status == "FAIL" and first_divergence is None:
            first_divergence = k

        label_str = (
            f"Joint {k}" if k <= table.n_joints else f"EE-{k - table.n_joints}"
        )
        print(
            f"  {label_str:<12}  {pe:<20.3e}  {oe:<18.3e}  {status}"
        )

    print(f"  {'-'*60}")
    print(f"  Max pos error : {max_pos_err:.3e} m   (tol {_POS_TOL:.0e} m)")
    print(f"  Max ori error : {max_ori_err:.3e} rad (tol {_ORI_TOL:.0e} rad)")

    if max_pos_err < _POS_TOL and max_ori_err < _ORI_TOL:
        print("  RESULT : PASS – all frames within tolerance.")
    else:
        print(f"  RESULT : FAIL – first divergence at frame {first_divergence}.")
        if first_divergence is not None:
            k = first_divergence
            print("\n  Diagnostic – first diverging frame:")
            print(f"    DH   T_{k}:\n{np.round(dh_chain[k], 8)}")
            print(f"    MJCF T_{k}:\n{np.round(mjcf_chain[dh_indices[k-1]], 8)}")
            diff = dh_chain[k] - mjcf_chain[dh_indices[k - 1]]
            print(f"    diff:\n{np.round(diff, 12)}")

    print(f"{'='*62}")
    return max_pos_err, max_ori_err
