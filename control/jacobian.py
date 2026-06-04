import numpy as np
from robot.kinematics import adjoint, matrix_exp_se3


def body_jacobian(theta: np.ndarray, B_list: list) -> np.ndarray:
    """
    Body Jacobian J_b(theta) for Body PoE.

    T(theta) = M exp(B1 theta1) ... exp(Bn thetan)

    Modern Robotics:
      Jb_n = Bn
      Jb_i = Ad_{exp(-B_{i+1} theta_{i+1}) ... exp(-Bn thetan)} B_i
    """
    theta = np.asarray(theta, dtype=float)
    n = len(B_list)
    Jb = np.zeros((6, n))

    T = np.eye(4)
    for i in reversed(range(n)):
        B = np.asarray(B_list[i], dtype=float)
        Jb[:, i] = adjoint(T) @ B
        T = T @ matrix_exp_se3(B, -theta[i])

    return Jb
