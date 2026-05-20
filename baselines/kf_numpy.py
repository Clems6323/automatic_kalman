"""NumPy reference Kalman filter — NOT differentiable, for validation only.

Do NOT import this module from anywhere under src/.
Verified to match automatic_kalman.filters.kf.KalmanFilter to 1e-10 (float64).
"""

import numpy as np
from numpy.typing import NDArray


class KalmanFilterNumpy:
    """Reference linear KF in pure NumPy.

    Uses the same Joseph-form covariance update and symmetrisation as the
    PyTorch implementation so the two can be compared step-for-step to 1e-10
    in float64.  ``np.linalg.solve`` is used instead of explicit inverse.

    Args:
        F: state transition matrix, shape (dim_x, dim_x)
        H: measurement matrix,      shape (dim_z, dim_x)
    """

    def __init__(
        self,
        F: NDArray[np.float64],
        H: NDArray[np.float64],
    ) -> None:
        self.F: NDArray[np.float64] = F.astype(np.float64)
        self.H: NDArray[np.float64] = H.astype(np.float64)
        self.dim_x: int = F.shape[0]
        self.dim_z: int = H.shape[0]

    def predict(
        self,
        x: NDArray[np.float64],
        P: NDArray[np.float64],
        Q: NDArray[np.float64],
        u: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict step: x = F x,  P = F P Fᵀ + Q."""
        x_pred: NDArray[np.float64] = self.F @ x
        P_pred: NDArray[np.float64] = self.F @ P @ self.F.T + Q
        return x_pred, P_pred

    def update(
        self,
        x: NDArray[np.float64],
        P: NDArray[np.float64],
        z: NDArray[np.float64],
        R: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Update step — returns (x_post, P_post, innovation y, innovation cov S)."""
        y: NDArray[np.float64] = z - self.H @ x
        PHT: NDArray[np.float64] = P @ self.H.T
        S: NDArray[np.float64] = self.H @ PHT + R

        # K = PHT @ S⁻¹  via solve so we match the spirit of cholesky_solve
        K: NDArray[np.float64] = np.linalg.solve(S.T, PHT.T).T

        x_post: NDArray[np.float64] = x + K @ y

        # Joseph form, symmetrised — mirrors the PyTorch implementation
        I = np.eye(self.dim_x, dtype=np.float64)
        I_KH: NDArray[np.float64] = I - K @ self.H
        P_post: NDArray[np.float64] = I_KH @ P @ I_KH.T + K @ R @ K.T
        P_post = 0.5 * (P_post + P_post.T)

        return x_post, P_post, y, S
