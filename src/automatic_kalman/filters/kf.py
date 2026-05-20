"""Linear Kalman filter."""

import torch
from torch import Tensor


class KalmanFilter:
    """Differentiable linear Kalman filter.

    Design rules (all enforced here):
    - Kalman gain computed via ``cholesky_solve``, never ``torch.inverse``.
    - Covariance update uses the Joseph form for numerical stability.
    - Covariances are symmetrised after every update.
    - Operates in float64; F and H are cast to match the input device/dtype.

    Args:
        F: state transition matrix, shape (dim_x, dim_x)
        H: measurement matrix,      shape (dim_z, dim_x)
    """

    def __init__(self, F: Tensor, H: Tensor) -> None:
        self.F: Tensor = F.double()
        self.H: Tensor = H.double()
        self.dim_x: int = F.shape[0]
        self.dim_z: int = H.shape[0]

    def predict(
        self,
        x: Tensor,
        P: Tensor,
        Q: Tensor,
        u: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict step.

        Args:
            x: prior state estimates,      (B, dim_x)
            P: prior covariances,          (B, dim_x, dim_x)
            Q: process noise covariance,   (dim_x, dim_x)
            u: unused (linear KF has no control input by default)

        Returns:
            x_pred: (B, dim_x)
            P_pred: (B, dim_x, dim_x)
        """
        F = self.F.to(dtype=x.dtype, device=x.device)

        x_pred = x @ F.mT                   # (B, dim_x)
        P_pred = F @ P @ F.mT + Q           # (B, dim_x, dim_x)
        return x_pred, P_pred

    def update(
        self,
        x: Tensor,
        P: Tensor,
        z: Tensor,
        R: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Update step.

        Kalman gain: K = P Hᵀ S⁻¹  via cholesky_solve (K.T = S⁻¹ P Hᵀ.T).
        Covariance:  Joseph form  (I − K H) P (I − K H)ᵀ + K R Kᵀ.

        Args:
            x: predicted states,             (B, dim_x)
            P: predicted covariances,        (B, dim_x, dim_x)
            z: measurements,                 (B, dim_z)
            R: measurement noise covariance, (dim_z, dim_z)

        Returns:
            x_post: (B, dim_x)
            P_post: (B, dim_x, dim_x)
            y:      innovations,             (B, dim_z)
            S:      innovation covariances,  (B, dim_z, dim_z)
        """
        H = self.H.to(dtype=x.dtype, device=x.device)

        y = z - x @ H.mT                    # (B, dim_z)  — innovation

        PHT = P @ H.mT                      # (B, dim_x, dim_z)
        S = H @ PHT + R                     # (B, dim_z, dim_z)  — innovation cov

        # K = P Hᵀ S⁻¹  ↔  K.T = S⁻¹ (P Hᵀ)ᵀ  →  solve S K.T = PHT.T
        L_S = torch.linalg.cholesky(S)
        K_T = torch.cholesky_solve(PHT.mT, L_S)          # (B, dim_z, dim_x)
        K = K_T.mT                          # (B, dim_x, dim_z)

        x_post = x + (K @ y.unsqueeze(-1)).squeeze(-1)   # (B, dim_x)

        # Joseph form — numerically stable, symmetric under gradient flow
        I = torch.eye(self.dim_x, dtype=x.dtype, device=x.device)
        I_KH = I - K @ H                    # (B, dim_x, dim_x)
        P_post = I_KH @ P @ I_KH.mT + K @ R @ K.mT      # (B, dim_x, dim_x)
        P_post = 0.5 * (P_post + P_post.mT)              # symmetrize

        return x_post, P_post, y, S
