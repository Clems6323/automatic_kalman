"""Cholesky-parameterised symmetric positive-definite covariance matrices."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# softplus⁻¹(1) so that the default factor() is exactly the identity matrix
_SOFTPLUS_INV_1: float = math.log(math.exp(1.0) - 1.0)


class CholeskyCovariance(nn.Module):
    """SPD covariance via a learned lower-triangular Cholesky factor L.

    The diagonal of L is kept strictly positive by passing the raw diagonal
    values through ``softplus``.  Off-diagonal entries are unconstrained.
    The covariance matrix is reconstructed as::

        cov = L @ L.T + eps * I

    where ``eps`` is a constant jitter (not tunable).

    Default initialisation: ``factor()`` = I, so ``matrix()`` = (1 + eps) * I.

    Args:
        dim: matrix dimension
        eps: jitter constant added to the diagonal for numerical stability
    """

    _L_raw: nn.Parameter

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Diagonal initialised to softplus⁻¹(1) so factor() starts as I
        L_init = torch.eye(dim, dtype=torch.float64) * _SOFTPLUS_INV_1
        self._L_raw = nn.Parameter(L_init)

    def factor(self) -> Tensor:
        """Return the lower-triangular Cholesky factor L.

        The diagonal is strictly positive (softplus); off-diagonals are free.
        Shape: (dim, dim).
        """
        L = self._L_raw.tril()
        off_diag = L.tril(-1)  # strictly lower-triangular part
        diag = torch.diag(F.softplus(L.diagonal()))  # (dim, dim) diagonal matrix
        return off_diag + diag

    def matrix(self) -> Tensor:
        """Return the full covariance matrix L @ L.T + eps * I.

        Always SPD and always in the autograd graph.
        Shape: (dim, dim).
        """
        L = self.factor()
        eps_I = self.eps * torch.eye(self.dim, dtype=L.dtype, device=L.device)
        return L @ L.T + eps_I
