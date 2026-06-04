"""
utils/lie_group.py
==================
SE(3) / se(3) 연산 유틸리티.

포함 함수:
  - matrix_log_se3 : log : SE(3) → ℝ^6  (vee of log T)
"""

import numpy as np
from robot.kinematics import skew


def matrix_log_se3(T: np.ndarray) -> np.ndarray:
    """
    Matrix logarithm of T ∈ SE(3).

    T = (R, p) 에 대해  log(T) ∈ se(3) 를 vee 연산으로 6-벡터로 반환.

    반환값: [ω̂θ ; vθ]

    수식 흐름 (Modern Robotics §3.3.3.2)
    ─────────────────────────────────────
    STEP 1 │ θ = arccos( (tr R − 1) / 2 )

    STEP 2 │ ω̂ 추출

      Case A  θ ≈ 0   →  순수 평행이동
                          ω̂θ = 0,  vθ = p

      Case B  θ ≈ π   →  수치 불안정 → (R+I)/2 = ω̂ω̂^T 이용

      Case C  일반    →  [ω̂ ] = (R − R^T) / (2 sinθ)       [Eq. 3.53]

    STEP 3 │ vθ 복원  (G(θ)v = p 로부터) [Eq. 3.91]

      θ·G^{-1}(θ) = I − (θ/2)[ω̂] + (1 − (θ/2)cot(θ/2))[ω̂]² [Eq. 3.92]

      vθ = θ·G^{-1}(θ)·p

    input
    ----------
    T : (4, 4)  SE(3) 변환 행렬

    output
    -------
    twist : (6,)  [ω̂θ ; vθ]

    ----------------
    matrix_log_se3(T) 자체는 T에 담긴 R이 항등행렬로부터 얼마나 회전했는가를 구한다. 근데, 어떤 T를 받느냐에 따라 달라지는데, trajectory.py를 보면,

    T_rel = T_inv(T_start) @ T_final   # ← 상대 변환
    log_rel = matrix_log_se3(T_rel)    # ← 이걸 log

    => T_start를 기준으로 T_final이 얼마나 다른가를 나타내는 상대 변환인 것.

    T_rel = T_start⁻¹ · T_final
    여기서 T_start = T_final이여서, T_rel이 I가 된다면, theta = 0 인 것임.

    또한 clik.py에서는

    T_err = T_inv(T_cur) @ T_des

    이런 형태로 넘겨줌. 여기서는 theta가 T_cur -> T_des까지의 회전 오차를 의미함. 전체 pose 오차.

    e_b[:3]  : 회전 오차  [ω̂θ]
    e_b[3:]  : 위치 오차  [vθ]


    즉, 여기서의 변환된 twist는 SE(3) 위에서의 오차 전체를 하나의 6D 벡터로 표현한 것.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R, p = T[:3, :3], T[:3, 3]

    # STEP 1: theta = arccos((tr R - 1) / 2)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # STEP 2: omega_hat 추출
    if theta < 1e-10:
        # Case A: theta ~ 0, pure translation
        return np.concatenate([np.zeros(3), p.copy()])

    if abs(theta - np.pi) < 1e-6:
        # Case B: theta ~ pi, (R+I)/2 = omega_hat @ omega_hat.T
        W = (R + np.eye(3)) / 2.0
        idx = int(np.argmax(np.diag(W)))
        omegahat = np.zeros(3)
        omegahat[idx] = np.sqrt(max(W[idx, idx], 0.0))
        for j in range(3):
            if j != idx:
                omegahat[j] = W[idx, j] / omegahat[idx]
    else:
        # Case C: general, [omega_hat] = (R - R^T) / (2 sin theta)
        log_R = (R - R.T) / (2.0 * np.sin(theta))
        omegahat = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])

    omega_theta = omegahat * theta  # angular part

    # STEP 3: v*theta 복원  G(theta)^{-1} * theta * p
    ws = skew(omegahat)
    tan_h = np.tan(theta / 2.0)
    coeff = 1.0 - (theta / 2.0) / tan_h if abs(tan_h) > 1e-10 else 1.0

    G_inv_theta = np.eye(3) - (theta / 2.0) * ws + coeff * (ws @ ws)
    v_theta = G_inv_theta @ p  # linear part

    return np.concatenate([omega_theta, v_theta])

    # 물체가 총 theta만큼 이동했을 때의 최종 트위스트는 각각의 성분에 거리를 곱한 [omega*theta . v*theta] 형태로 표현됩니다. 따라서 반환값은 [omega*theta ; v*theta] 형태의 6D 벡터
