"""
robot/kinematics.py
===================
수학 유틸리티 + Forward Kinematics 

포함 함수:
  유틸리티
    skew           : ℝ^3 → so(3)
    matrix_exp_se3 : se(3) × ℝ → SE(3)   Rodrigues
    adjoint        : SE(3) → ℝ^{6×6}     Ad_T
    T_inv          : SE(3) → SE(3)        T^{-1}

  Forward Kinematics
    space_poe_fk   : Space PoE  T = exp([S1]θ1)···exp([Sn]θn)·M
    body_poe_fk    : Body  PoE  T = M·exp([B1]θ1)···exp([Bn]θn)
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────
#  수학 유틸리티
# ──────────────────────────────────────────────────────────────────

def skew(w: np.ndarray) -> np.ndarray:
    """
    3-vector → 3×3 skew-symmetric matrix.

    [w]_× = [[  0, -w3,  w2],
              [ w3,   0, -w1],
              [-w2,  w1,   0]]

    Parameters
    ----------
    w : (3,)

    Returns
    -------
    W : (3, 3)  skew-symmetric
    """
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def matrix_exp_se3(S: np.ndarray, theta: float) -> np.ndarray:
    """
    Matrix exponential  e^{[S]θ} ∈ SE(3).   (Prop 3.25, Modern Robotics)

    Revolute  (‖ω‖ ≠ 0):
      R = I + (sin‖ω‖θ / ‖ω‖)[ω] + ((1−cos‖ω‖θ) / ‖ω‖²)[ω]²
      p = G·v
      G = Iθ + ((1−cos‖ω‖θ) / ‖ω‖²)[ω] + ((‖ω‖θ−sin‖ω‖θ) / ‖ω‖³)[ω]²

    Prismatic (‖ω‖ ≈ 0):
      R = I,  p = vθ

    Parameters
    ----------
    S     : (6,)   screw axis [ω; v]
    theta : float  joint variable

    Returns
    -------
    T : (4, 4)  SE(3)
    """
    S = np.asarray(S, dtype=float).reshape(6)
    w = S[:3]
    v = S[3:]
    w_norm = np.linalg.norm(w)

    T = np.eye(4)

    if w_norm < 1e-12:
        T[:3, 3] = v * theta
        return T

    W = skew(w)
    W2 = W @ W
    wt = w_norm * theta

    R = (
        np.eye(3)
        + (np.sin(wt) / w_norm) * W
        + ((1.0 - np.cos(wt)) / (w_norm**2)) * W2
    )
    G = (
        np.eye(3) * theta
        + ((1.0 - np.cos(wt)) / (w_norm**2)) * W
        + ((wt - np.sin(wt)) / (w_norm**3)) * W2
    )

    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


def adjoint(T: np.ndarray) -> np.ndarray:
    """
    Adjoint representation  [Ad_T] ∈ ℝ^{6×6}.   (Def 3.20)

    [Ad_T] = [[ R,    0  ],
               [ [p]R, R  ]]

    Parameters
    ----------
    T : (4, 4)  SE(3)

    Returns
    -------
    Ad : (6, 6)
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]

    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def T_inv(T: np.ndarray) -> np.ndarray:
    """
    SE(3) inverse.

    T^{-1} = [[ R^T,  -R^T p ],
               [  0,     1   ]]

    Parameters
    ----------
    T : (4, 4)  SE(3)

    Returns
    -------
    Ti : (4, 4)  SE(3)
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]

    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


# ──────────────────────────────────────────────────────────────────
#  Forward Kinematics
# ──────────────────────────────────────────────────────────────────

def space_poe_fk(q: np.ndarray, S_list: list, M: np.ndarray) -> np.ndarray:
    """
    Space Product of Exponentials FK.

    T(θ) = e^{[S1]θ1} · e^{[S2]θ2} · … · e^{[Sn]θn} · M

    Parameters
    ----------
    q      : (n,)   joint angles [rad]
    S_list : list of (6,)  space screw axes
    M      : (4, 4) home configuration

    Returns
    -------
    T : (4, 4)  end-effector pose in world frame
    """
    T = np.eye(4)
    for S, theta in zip(S_list, q):
        T = T @ matrix_exp_se3(S, theta)
    return T @ M


def body_poe_fk(q: np.ndarray, B_list: list, M: np.ndarray) -> np.ndarray:
    """
    Body Product of Exponentials FK.

    T(θ) = M · e^{[B1]θ1} · e^{[B2]θ2} · … · e^{[Bn]θn}

    Parameters
    ----------
    q      : (n,)   joint angles [rad]
    B_list : list of (6,)  body screw axes
    M      : (4, 4) home configuration

    Returns
    -------
    T : (4, 4)  end-effector pose in world frame
    """
    T = np.array(M, dtype=float, copy=True)
    for B, theta in zip(B_list, q):
        T = T @ matrix_exp_se3(B, theta)
    return T
