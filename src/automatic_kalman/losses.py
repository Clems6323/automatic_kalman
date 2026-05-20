"""Consistency losses for Kalman filter covariance learning.

Both ``nis_consistency`` and ``nees_consistency`` share the same
``(FilterOutputs, ...) -> scalar`` signature so the training loop is
loss-agnostic.

The log-det penalty prevents the trivial minimiser of inflating R/P to
drive the Mahalanobis metric to zero — without it, the optimiser learns
to over-inflate covariances and NIS/NEES → 0 rather than → dim_z/dim_x.
"""

import torch
from torch import Tensor

from automatic_kalman.types import FilterOutputs


def nis_consistency(outputs: FilterOutputs, lam: float = 1.0) -> Tensor:
    """NIS consistency loss (use when ground-truth state is unavailable).

    Minimises::

        (mean_NIS - dim_z)² + λ · mean(log det S)

    where NIS_k = yᵀ S⁻¹ y is computed via ``cholesky_solve`` to keep
    the full autograd graph intact (no explicit matrix inverse).

    Under a correctly tuned filter, ``mean_NIS`` converges to ``dim_z``
    and the log-det term prevents covariance inflation.

    Args:
        outputs: filter outputs from one T-step pass
        lam:     weight on the log-det regulariser

    Returns:
        scalar Tensor with gradient to ``L_Q`` and ``L_R``
    """
    y = outputs.innovations        # (B, T, dim_z)
    S = outputs.innovation_covs    # (B, T, dim_z, dim_z)
    dim_z = float(y.shape[-1])

    L_S = torch.linalg.cholesky(S)                                                       # (B, T, dim_z, dim_z)
    S_inv_y = torch.cholesky_solve(y.unsqueeze(-1), L_S).squeeze(-1)                     # (B, T, dim_z)
    nis = (y * S_inv_y).sum(-1)                                           # (B, T)

    # log det S = 2 * Σ log(diag(L_S))
    log_det_S = 2.0 * L_S.diagonal(dim1=-2, dim2=-1).log().sum(-1)       # (B, T)

    return (nis.mean() - dim_z) ** 2 + lam * log_det_S.mean()


def nees_consistency(outputs: FilterOutputs, lam: float = 1.0) -> Tensor:
    """NEES consistency loss (use when ground-truth state is available).

    Minimises::

        (mean_NEES - dim_x)² + λ · mean(log det P)

    where NEES_k = eᵀ P⁻¹ e,  e = x_post − x_true.

    Args:
        outputs: filter outputs including ``states_true``
        lam:     weight on the log-det regulariser

    Returns:
        scalar Tensor with gradient to ``L_Q`` and ``L_R``

    Raises:
        ValueError: if ``outputs.states_true`` is None
    """
    if outputs.states_true is None:
        raise ValueError(
            "nees_consistency requires outputs.states_true — "
            "use nis_consistency when ground truth is unavailable."
        )

    x = outputs.states              # (B, T, dim_x)
    x_true = outputs.states_true    # (B, T, dim_x)
    P = outputs.state_covs          # (B, T, dim_x, dim_x)
    dim_x = float(x.shape[-1])

    e = x - x_true                                                         # (B, T, dim_x)
    L_P = torch.linalg.cholesky(P)                                                       # (B, T, dim_x, dim_x)
    P_inv_e = torch.cholesky_solve(e.unsqueeze(-1), L_P).squeeze(-1)                     # (B, T, dim_x)
    nees = (e * P_inv_e).sum(-1)                                           # (B, T)

    log_det_P = 2.0 * L_P.diagonal(dim1=-2, dim2=-1).log().sum(-1)        # (B, T)

    return (nees.mean() - dim_x) ** 2 + lam * log_det_P.mean()
