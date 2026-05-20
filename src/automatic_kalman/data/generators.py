"""Synthetic trajectory generators for filter training and validation."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.func
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class TrajectoryBatch:
    """A batch of simulated trajectories.

    Attributes:
        measurements: noisy observations,   shape (B, T, dim_z)
        states:        ground-truth states,  shape (B, T, dim_x)
    """

    measurements: Tensor
    states: Tensor


class LinearGaussianGenerator:
    """Generate batched trajectories from a linear-Gaussian system.

    Dynamics (discrete-time)::

        x_{k+1} = F x_k + w_k,    w_k ~ N(0, Q_true)
        z_k     = H x_k + v_k,    v_k ~ N(0, R_true)

    Noise is sampled via the Cholesky factors ``L_Q``, ``L_R`` so that
    ``w = L_Q @ z,  z ~ N(0, I)`` gives the correct covariance.
    All outputs are ``float64``.

    Args:
        F:      state transition matrix, shape (dim_x, dim_x)
        H:      measurement matrix,      shape (dim_z, dim_x)
        Q_true: true process noise covariance,     shape (dim_x, dim_x)
        R_true: true measurement noise covariance, shape (dim_z, dim_z)
        x0:     initial state, shape (dim_x,)
    """

    def __init__(
        self,
        F: Tensor,
        H: Tensor,
        Q_true: Tensor,
        R_true: Tensor,
        x0: Tensor,
    ) -> None:
        self.F = F.double()
        self.H = H.double()
        self.Q_true = Q_true.double()
        self.R_true = R_true.double()
        self.x0 = x0.double()
        self.dim_x: int = F.shape[0]
        self.dim_z: int = H.shape[0]
        self._L_Q = torch.linalg.cholesky(self.Q_true)
        self._L_R = torch.linalg.cholesky(self.R_true)

    def __call__(self, B: int, T: int) -> TrajectoryBatch:
        """Sample ``B`` independent trajectories of length ``T``.

        Args:
            B: batch size (number of independent realisations)
            T: trajectory length (number of time steps)

        Returns:
            :class:`TrajectoryBatch` with ``measurements`` (B, T, dim_z)
            and ``states`` (B, T, dim_x).
        """
        with torch.no_grad():
            states: list[Tensor] = []
            x = self.x0.unsqueeze(0).expand(B, -1).clone()  # (B, dim_x)

            for _ in range(T):
                states.append(x.clone())
                # w ~ N(0, Q_true): sample z ~ N(0, I) then w = z @ L_Q.T
                w = torch.randn(B, self.dim_x, dtype=torch.float64) @ self._L_Q.T
                x = x @ self.F.T + w

            stacked_states = torch.stack(states, dim=1)  # (B, T, dim_x)

            # z = H x + v,  v ~ N(0, R_true)
            v = torch.randn(B, T, self.dim_z, dtype=torch.float64) @ self._L_R.T
            # (B, T, dim_x, 1) -> (B, T, dim_z, 1) -> (B, T, dim_z)
            measurements = (self.H @ stacked_states.unsqueeze(-1)).squeeze(-1) + v

        return TrajectoryBatch(measurements=measurements, states=stacked_states)


class NonlinearGenerator:
    """Generate batched trajectories from a nonlinear dynamical system.

    Dynamics (discrete-time)::

        x_{k+1} = f(x_k, None) + w_k,   w_k ~ N(0, Q_true)
        z_k     = h(x_k) + v_k,          v_k ~ N(0, R_true)

    ``f`` and ``h`` are **unbatched** callables (single state vector).
    Batching is handled internally via ``torch.func.vmap``.

    Args:
        f:      state transition, single-sample: ``(dim_x,) → (dim_x,)``
        h:      measurement function, single-sample: ``(dim_x,) → (dim_z,)``
        Q_true: true process noise covariance,     shape (dim_x, dim_x)
        R_true: true measurement noise covariance, shape (dim_z, dim_z)
        x0:     initial state, shape (dim_x,)
    """

    def __init__(
        self,
        f: Callable[[Tensor, None], Tensor],
        h: Callable[[Tensor], Tensor],
        Q_true: Tensor,
        R_true: Tensor,
        x0: Tensor,
    ) -> None:
        self.f = f
        self.h = h
        self.Q_true = Q_true.double()
        self.R_true = R_true.double()
        self.x0 = x0.double()
        self.dim_x: int = x0.shape[0]
        self.dim_z: int = R_true.shape[0]
        self._L_Q = torch.linalg.cholesky(self.Q_true)
        self._L_R = torch.linalg.cholesky(self.R_true)
        self._f_vmapped = torch.func.vmap(lambda xi: self.f(xi, None))
        self._h_vmapped = torch.func.vmap(self.h)

    def __call__(self, B: int, T: int) -> "TrajectoryBatch":
        """Sample ``B`` independent trajectories of length ``T``."""
        with torch.no_grad():
            states: list[Tensor] = []
            x = self.x0.unsqueeze(0).expand(B, -1).clone()

            for _ in range(T):
                states.append(x.clone())
                w = torch.randn(B, self.dim_x, dtype=torch.float64) @ self._L_Q.T
                x = self._f_vmapped(x) + w

            stacked_states = torch.stack(states, dim=1)  # (B, T, dim_x)

            v = torch.randn(B, T, self.dim_z, dtype=torch.float64) @ self._L_R.T
            meas_rows = [
                self._h_vmapped(stacked_states[:, t, :]) for t in range(T)
            ]
            measurements = torch.stack(meas_rows, dim=1) + v  # (B, T, dim_z)

        return TrajectoryBatch(measurements=measurements, states=stacked_states)


class TrajectoryDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch Dataset wrapping a :class:`TrajectoryBatch`.

    Each item is a ``(measurements, states)`` pair for a single trajectory.
    Compatible with ``torch.utils.data.DataLoader`` out of the box.

    Example::

        gen = LinearGaussianGenerator(F, H, Q, R, x0)
        dataset = TrajectoryDataset(gen(B=200, T=50))
        loader = DataLoader(dataset, batch_size=20, shuffle=True)
    """

    def __init__(self, batch: TrajectoryBatch) -> None:
        self._measurements = batch.measurements  # (B, T, dim_z)
        self._states = batch.states              # (B, T, dim_x)

    def __len__(self) -> int:
        return self._measurements.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self._measurements[idx], self._states[idx]
