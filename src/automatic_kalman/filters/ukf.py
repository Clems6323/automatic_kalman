"""Unscented Kalman filter (Merwe scaled sigma points)."""

import math
from collections.abc import Callable

import torch
import torch.func
from torch import Tensor


def _safe_cholesky(M: Tensor, jitter: float = 1e-8) -> Tensor:
    """Batched Cholesky with one jitter retry on failure (CLAUDE.md rule 4)."""
    try:
        return torch.linalg.cholesky(M)
    except RuntimeError:
        eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
        try:
            return torch.linalg.cholesky(M + jitter * eye)
        except RuntimeError:
            raise RuntimeError(
                f"Cholesky failed with jitter {jitter:.1e}; "
                "P may not be PSD — check initialisation and gradient clipping."
            )


class UnscentedKalmanFilter:
    """Differentiable UKF with Merwe scaled sigma points.

    Sigma-point generation uses ``torch.linalg.cholesky(P)`` which is
    differentiable in modern PyTorch.  ``alpha``, ``beta``, ``kappa`` are
    fixed during Q/R training — do NOT include them in the optimizer.

    The Filter protocol splits predict and update.  The update step
    regenerates sigma points from the predicted ``(x, P)`` via Cholesky,
    which is the standard "two-step" UKF formulation.

    For linear ``f`` and ``h`` the UKF is algebraically equivalent to the
    classic KF (verified in the test suite).

    Args:
        f:     state transition, single-sample: ``(dim_x,) → (dim_x,)``
        h:     measurement function, single-sample: ``(dim_x,) → (dim_z,)``
        dim_x: state dimension
        dim_z: measurement dimension
        alpha: sigma-point spread (default 1e-3)
        beta:  prior knowledge of distribution (default 2, optimal for Gaussian)
        kappa: secondary scaling (default 0)
    """

    def __init__(
        self,
        f: Callable[[Tensor, Tensor | None], Tensor],
        h: Callable[[Tensor], Tensor],
        dim_x: int,
        dim_z: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.f = f
        self.h = h
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        n = dim_x
        lam: float = alpha ** 2 * (n + kappa) - n
        self._scale: float = math.sqrt(n + lam)     # sqrt(n+λ) — scales Cholesky cols

        # W_m: mean weights, shape (2n+1,)
        w_i = 1.0 / (2.0 * (n + lam))
        W_m = torch.full((2 * n + 1,), w_i, dtype=torch.float64)
        W_m[0] = lam / (n + lam)

        # W_c: covariance weights — identical to W_m except index 0 gets β correction
        W_c = W_m.clone()
        W_c[0] = lam / (n + lam) + (1.0 - alpha ** 2 + beta)

        self._W_m = W_m   # (2n+1,)
        self._W_c = W_c   # (2n+1,)

        # Sanity: mean weights must sum to 1
        assert abs(W_m.sum().item() - 1.0) < 1e-10, "W_m does not sum to 1"
        # Covariance weights differ at index 0
        assert abs(W_c[0].item() - W_m[0].item()) > 1e-12, "W_c[0] must differ from W_m[0]"

    # ── internal helpers ──────────────────────────────────────────────────────

    def _sigma_points(self, x: Tensor, P: Tensor) -> Tensor:
        """Generate (B, 2·dim_x+1, dim_x) sigma points.

        X_0     = x
        X_{1:n} = x + scale · L.T[i]    (i-th row of L.T = i-th col of L)
        X_{n+1:} = x − scale · L.T[i]
        """
        L = _safe_cholesky(P)                                  # (B, n, n)
        cols = self._scale * L.mT                              # (B, n, n): row i = col i of L
        X_plus  = x.unsqueeze(1) + cols                        # (B, n, n)
        X_minus = x.unsqueeze(1) - cols                        # (B, n, n)
        return torch.cat([x.unsqueeze(1), X_plus, X_minus], dim=1)  # (B, 2n+1, n)

    # ── Filter protocol ───────────────────────────────────────────────────────

    def predict(
        self,
        x: Tensor,
        P: Tensor,
        Q: Tensor,
        u: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """UKF predict: propagate sigma points through f, reconstruct mean/cov.

        Args:
            x: prior state estimates,      (B, dim_x)
            P: prior covariances,          (B, dim_x, dim_x)
            Q: process noise covariance,   (dim_x, dim_x)
            u: optional control inputs,    (B, dim_u) or None

        Returns:
            x_pred: (B, dim_x)
            P_pred: (B, dim_x, dim_x)
        """
        W_m = self._W_m.to(dtype=x.dtype, device=x.device)   # (2n+1,)
        W_c = self._W_c.to(dtype=x.dtype, device=x.device)   # (2n+1,)
        n_sig = 2 * self.dim_x + 1

        sigmas = self._sigma_points(x, P)   # (B, 2n+1, n)

        # Propagate each sigma point through f
        if u is None:
            sigmas_f = torch.func.vmap(
                torch.func.vmap(lambda xi: self.f(xi, None))
            )(sigmas)
        else:
            # Expand u over sigma-point dimension so vmap can map jointly
            u_exp = u.unsqueeze(1).expand(-1, n_sig, -1)   # (B, 2n+1, dim_u)
            sigmas_f = torch.func.vmap(
                torch.func.vmap(self.f)
            )(sigmas, u_exp)

        # Predicted mean: x_pred[b] = Σ_i W_m[i] · sigmas_f[b, i]
        x_pred = (W_m.view(1, -1, 1) * sigmas_f).sum(1)    # (B, n)

        # Predicted covariance: P_pred = Σ_i W_c[i] · δ_i δ_i^T + Q
        delta = sigmas_f - x_pred.unsqueeze(1)              # (B, 2n+1, n)
        P_pred = torch.einsum("bsi,s,bsj->bij", delta, W_c, delta) + Q
        P_pred = 0.5 * (P_pred + P_pred.mT)

        return x_pred, P_pred

    def update(
        self,
        x: Tensor,
        P: Tensor,
        z: Tensor,
        R: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """UKF update: regenerate sigma points from predicted (x, P), transform via h.

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
        W_m = self._W_m.to(dtype=x.dtype, device=x.device)   # (2n+1,)
        W_c = self._W_c.to(dtype=x.dtype, device=x.device)   # (2n+1,)

        sigmas = self._sigma_points(x, P)   # (B, 2n+1, n)

        # Transform each sigma point through h
        sigmas_z = torch.func.vmap(
            torch.func.vmap(self.h)
        )(sigmas)                                              # (B, 2n+1, m)

        # Predicted measurement mean
        z_pred = (W_m.view(1, -1, 1) * sigmas_z).sum(1)     # (B, m)
        y = z - z_pred                                        # innovation (B, m)

        # Deviations from predicted means (used for covariance terms)
        delta_x = sigmas - x.unsqueeze(1)                    # (B, 2n+1, n)
        delta_z = sigmas_z - z_pred.unsqueeze(1)             # (B, 2n+1, m)

        # Innovation covariance S and cross-covariance P_xz
        S   = torch.einsum("bsi,s,bsj->bij", delta_z, W_c, delta_z) + R  # (B, m, m)
        P_xz = torch.einsum("bsi,s,bsj->bij", delta_x, W_c, delta_z)     # (B, n, m)

        # Kalman gain: K = P_xz S^{-1}  via cholesky_solve
        L_S = torch.linalg.cholesky(S)
        K   = torch.cholesky_solve(P_xz.mT, L_S).mT         # (B, n, m)

        x_post = x + (K @ y.unsqueeze(-1)).squeeze(-1)       # (B, n)

        # Covariance update (standard UKF form — algebraically equal to Joseph
        # form for linear h, symmetrized for gradient stability)
        P_post = P - K @ S @ K.mT
        P_post = 0.5 * (P_post + P_post.mT)

        return x_post, P_post, y, S
