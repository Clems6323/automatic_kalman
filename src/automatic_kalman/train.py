"""Filter-agnostic training loop for covariance learning."""

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from automatic_kalman.covariance import CholeskyCovariance
from automatic_kalman.filters.base import Filter
from automatic_kalman.types import FilterOutputs

_log = logging.getLogger(__name__)


def train(
    filter: Filter,
    cov_Q: CholeskyCovariance,
    cov_R: CholeskyCovariance,
    loss_fn: Callable[[FilterOutputs], Tensor],
    loader: DataLoader[Any],
    *,
    x0: Tensor,
    P0: Tensor,
    epochs: int = 100,
    lr: float = 1e-2,
    grad_clip: float = 10.0,
    cond_warn_threshold: float = 1e8,
) -> list[float]:
    """Train Q and R covariances for any Filter satisfying the Filter protocol.

    The loop never branches on filter type — ``filter`` is treated as an
    opaque object that implements ``predict`` and ``update``.

    Gradient clipping (norm ≤ ``grad_clip``) is applied to the Cholesky
    factors of ``cov_Q`` and ``cov_R`` to handle spikes near singular S.

    Condition numbers of S and P are logged at DEBUG level each epoch; a
    jump above ``cond_warn_threshold`` triggers a WARNING.

    Device is inferred from ``cov_Q`` parameters — move ``cov_Q`` and
    ``cov_R`` to CUDA before calling to enable GPU training.

    Args:
        filter:              object satisfying the Filter protocol
        cov_Q:               CholeskyCovariance for process noise Q
        cov_R:               CholeskyCovariance for measurement noise R
        loss_fn:             ``FilterOutputs -> scalar Tensor``, e.g. ``nis_consistency``
        loader:              yields ``(z_batch, x_true_batch)`` or ``(z_batch,)`` tuples;
                             tensors are cast to float64 and moved to device internally
        x0:                  initial state estimate, shape ``(dim_x,)``
        P0:                  initial covariance,      shape ``(dim_x, dim_x)``
        epochs:              number of full passes over the dataset
        lr:                  Adam learning rate
        grad_clip:           max gradient norm applied to L_Q and L_R
        cond_warn_threshold: emit WARNING when cond(S) or cond(P) exceeds this value

    Returns:
        Per-epoch average loss as a list of Python floats.
    """
    params = list(cov_Q.parameters()) + list(cov_R.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    epoch_losses: list[float] = []

    device = next(iter(cov_Q.parameters())).device
    x0_ = x0.detach().double().to(device)
    P0_ = P0.detach().double().to(device)

    for epoch in range(epochs):
        batch_losses: list[float] = []
        last_S: Tensor | None = None
        last_P: Tensor | None = None

        for batch in loader:
            # Support loaders that yield (z,) or (z, x_true)
            z_batch: Tensor
            x_true_batch: Tensor | None

            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                z_batch, x_true_batch = batch[0], batch[1]
                x_true_batch = x_true_batch.double().to(device)
            else:
                z_batch = batch[0] if isinstance(batch, (list, tuple)) else batch
                x_true_batch = None

            z_batch = z_batch.double().to(device)  # (B, T, dim_z)
            B, T, _ = z_batch.shape

            optimizer.zero_grad()

            # Build Q and R from the learnable Cholesky factors
            Q = cov_Q.matrix()  # (dim_x, dim_x) — in the autograd graph
            R = cov_R.matrix()  # (dim_z, dim_z)

            # Broadcast initial state/cov across batch dimension
            x: Tensor = x0_.unsqueeze(0).expand(B, -1).clone()          # (B, dim_x)
            P: Tensor = P0_.unsqueeze(0).expand(B, -1, -1).clone()      # (B, dim_x, dim_x)

            innovations_list: list[Tensor] = []
            innovation_covs_list: list[Tensor] = []
            states_list: list[Tensor] = []
            state_covs_list: list[Tensor] = []

            for t in range(T):
                z_t = z_batch[:, t, :]                              # (B, dim_z)
                x, P = filter.predict(x, P, Q, None)
                x, P, y, S = filter.update(x, P, z_t, R)

                innovations_list.append(y)      # (B, dim_z)
                innovation_covs_list.append(S)  # (B, dim_z, dim_z)
                states_list.append(x)           # (B, dim_x)
                state_covs_list.append(P)       # (B, dim_x, dim_x)

            last_S, last_P = S, P

            outputs = FilterOutputs(
                innovations=torch.stack(innovations_list, dim=1),         # (B, T, dim_z)
                innovation_covs=torch.stack(innovation_covs_list, dim=1), # (B, T, dim_z, dim_z)
                states=torch.stack(states_list, dim=1),                   # (B, T, dim_x)
                state_covs=torch.stack(state_covs_list, dim=1),           # (B, T, dim_x, dim_x)
                states_true=x_true_batch,
            )

            loss = loss_fn(outputs)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            batch_losses.append(loss.item())

        epoch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float("nan")
        epoch_losses.append(epoch_loss)

        # Condition-number diagnostics (float64 guard: no_grad, detach)
        if last_S is not None and last_P is not None:
            with torch.no_grad():
                cond_S = torch.linalg.cond(last_S.detach().mean(0)).item()
                cond_P = torch.linalg.cond(last_P.detach().mean(0)).item()
            _log.debug(
                "epoch %d  loss=%.6f  cond(S)=%.2e  cond(P)=%.2e",
                epoch, epoch_loss, cond_S, cond_P,
            )
            if cond_S > cond_warn_threshold or cond_P > cond_warn_threshold:
                _log.warning(
                    "epoch %d: high condition number — cond(S)=%.2e  cond(P)=%.2e"
                    " — possible instability, check Q/R initialisation",
                    epoch, cond_S, cond_P,
                )
        else:
            _log.debug("epoch %d  loss=%.6f", epoch, epoch_loss)

    return epoch_losses
