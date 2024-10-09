import numpy as np


def safe_inverse_ps_matrix(matrix: np.ndarray, eps: float = 10**-6) -> np.ndarray:
    """Compute the inverse of symmetric positive definite matrix.
    Correct np.linalg.inv implementation when matrix has bad conditionning number.

    Inversion of eigenvalues < eps will be heavily perturbed.
    This function is still work in progress
    """

    return np.linalg.inv(0.5 * (matrix + matrix.T) + eps * np.eye(len(matrix)))
