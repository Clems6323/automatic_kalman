"""Filter protocol shared by KF, EKF, and UKF."""

from typing import Protocol

from torch import Tensor


class Filter(Protocol):
    """Common interface for all differentiable Kalman-family filters.

    The training loop and losses never branch on filter type — every filter
    must satisfy this protocol exactly.

    All methods operate on batched inputs so the training loop vectorises
    over the batch dimension B without branching::

        x : (B, dim_x)          — batch of state estimates
        P : (B, dim_x, dim_x)   — batch of covariance matrices
        Q : (dim_x, dim_x)      — shared process noise (single learned matrix)
        R : (dim_z, dim_z)      — shared measurement noise (single learned matrix)
        z : (B, dim_z)          — batch of measurements at one time step
    """

    def predict(
        self,
        x: Tensor,
        P: Tensor,
        Q: Tensor,
        u: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Propagate state and covariance forward one step.

        Args:
            x: prior state estimates,       shape (B, dim_x)
            P: prior covariances,           shape (B, dim_x, dim_x)
            Q: process noise covariance,    shape (dim_x, dim_x)
            u: optional control inputs,     shape (B, dim_u) or None

        Returns:
            x_pred: predicted states,       (B, dim_x)
            P_pred: predicted covariances,  (B, dim_x, dim_x)
        """
        ...

    def update(
        self,
        x: Tensor,
        P: Tensor,
        z: Tensor,
        R: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Incorporate a batch of measurements.

        Args:
            x: predicted states,             shape (B, dim_x)
            P: predicted covariances,        shape (B, dim_x, dim_x)
            z: measurements,                 shape (B, dim_z)
            R: measurement noise covariance, shape (dim_z, dim_z)

        Returns:
            x_post: posterior states,        (B, dim_x)
            P_post: posterior covariances,   (B, dim_x, dim_x)
            y:      innovations z - H x_pred,(B, dim_z)
            S:      innovation covariances,  (B, dim_z, dim_z)
        """
        ...
