"""Extended Kalman filter."""

from collections.abc import Callable

import torch
import torch.func
from torch import Tensor


class ExtendedKalmanFilter:
    """Differentiable EKF with autodiff Jacobians via ``torch.func.jacrev``.

    The user supplies ``f`` and ``h`` as **unbatched** callables (single state
    vector).  Batching and Jacobian computation are handled internally via
    ``torch.func.vmap`` and ``torch.func.jacrev``.

    Design rules (same as KF):
    - Kalman gain via ``torch.cholesky_solve``.
    - Joseph-form covariance update.
    - Covariances symmetrised after every predict and update.
    - Operates in float64.

    Args:
        f:     state transition, single-sample: ``(dim_x,) → (dim_x,)``
        h:     measurement function, single-sample: ``(dim_x,) → (dim_z,)``
        dim_x: state dimension
        dim_z: measurement dimension
        F_fn:  optional analytic Jacobian of f, ``(dim_x,) → (dim_x, dim_x)``
        H_fn:  optional analytic Jacobian of h, ``(dim_x,) → (dim_z, dim_x)``
    """

    def __init__(
        self,
        f: Callable[[Tensor, Tensor | None], Tensor],
        h: Callable[[Tensor], Tensor],
        dim_x: int,
        dim_z: int,
        F_fn: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        H_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        self.f = f
        self.h = h
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.F_fn = F_fn
        self.H_fn = H_fn

    # ── internal helpers ──────────────────────────────────────────────────────

    def _jac_f(self, x: Tensor, u: Tensor | None) -> Tensor:
        """Batched Jacobian ∂f/∂x.  Returns (B, dim_x, dim_x)."""
        if self.F_fn is not None:
            if u is None:
                return torch.func.vmap(lambda xi: self.F_fn(xi, None))(x)
            return torch.func.vmap(self.F_fn)(x, u)
        if u is None:
            return torch.func.vmap(
                torch.func.jacrev(lambda xi: self.f(xi, None))
            )(x)
        return torch.func.vmap(
            torch.func.jacrev(self.f, argnums=0)
        )(x, u)

    def _jac_h(self, x: Tensor) -> Tensor:
        """Batched Jacobian ∂h/∂x.  Returns (B, dim_z, dim_x)."""
        if self.H_fn is not None:
            return torch.func.vmap(self.H_fn)(x)
        return torch.func.vmap(torch.func.jacrev(self.h))(x)

    # ── Filter protocol ───────────────────────────────────────────────────────

    def predict(
        self,
        x: Tensor,
        P: Tensor,
        Q: Tensor,
        u: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """EKF predict.

        Propagates the mean via the nonlinear ``f`` and the covariance via the
        Jacobian ``F = ∂f/∂x`` evaluated at the current estimate.

        Args:
            x: prior state estimates,      (B, dim_x)
            P: prior covariances,          (B, dim_x, dim_x)
            Q: process noise covariance,   (dim_x, dim_x)
            u: optional control inputs,    (B, dim_u) or None

        Returns:
            x_pred: (B, dim_x)
            P_pred: (B, dim_x, dim_x)
        """
        # Propagate mean through nonlinear f
        if u is None:
            x_pred = torch.func.vmap(lambda xi: self.f(xi, None))(x)
        else:
            x_pred = torch.func.vmap(self.f)(x, u)

        # Linearise around x to propagate covariance
        F = self._jac_f(x, u)                            # (B, dim_x, dim_x)
        P_pred = F @ P @ F.mT + Q                        # (B, dim_x, dim_x)
        P_pred = 0.5 * (P_pred + P_pred.mT)
        return x_pred, P_pred

    def update(
        self,
        x: Tensor,
        P: Tensor,
        z: Tensor,
        R: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """EKF update.

        Linearises ``h`` at the predicted state, then applies the standard
        Kalman update with Joseph-form covariance.

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
        z_pred = torch.func.vmap(self.h)(x)              # (B, dim_z)
        y = z - z_pred

        H = self._jac_h(x)                               # (B, dim_z, dim_x)
        PHT = P @ H.mT                                   # (B, dim_x, dim_z)
        S = H @ PHT + R                                  # (B, dim_z, dim_z)

        L_S = torch.linalg.cholesky(S)
        K = torch.cholesky_solve(PHT.mT, L_S).mT        # (B, dim_x, dim_z)

        x_post = x + (K @ y.unsqueeze(-1)).squeeze(-1)  # (B, dim_x)

        I = torch.eye(self.dim_x, dtype=x.dtype, device=x.device)
        I_KH = I - K @ H
        P_post = I_KH @ P @ I_KH.mT + K @ R @ K.mT
        P_post = 0.5 * (P_post + P_post.mT)

        return x_post, P_post, y, S
