"""Shared data structures passed between the filter, training loop, and losses."""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class FilterOutputs:
    """Collected filter outputs for one full T-step pass over a batch.

    All tensors carry gradients — do not detach them before passing to a loss.

    Attributes:
        innovations:     per-step innovations y = z - H x_pred,     (B, T, dim_z)
        innovation_covs: per-step innovation covariances S,          (B, T, dim_z, dim_z)
        states:          per-step posterior state estimates,         (B, T, dim_x)
        state_covs:      per-step posterior covariances P,           (B, T, dim_x, dim_x)
        states_true:     ground-truth states when available,         (B, T, dim_x) or None
    """

    innovations: Tensor
    innovation_covs: Tensor
    states: Tensor
    state_covs: Tensor
    states_true: Tensor | None = None
