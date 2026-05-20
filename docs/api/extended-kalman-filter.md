# ExtendedKalmanFilter

```python
from automatic_kalman.filters.ekf import ExtendedKalmanFilter
```

## Overview

The Extended Kalman filter for nonlinear systems.  The user supplies
`f` (state transition) and `h` (measurement) as **unbatched** PyTorch
callables; batching and Jacobian computation are handled internally via
`torch.func.vmap` and `torch.func.jacrev`.

This is the key design advantage over a naive EKF: the user writes
clean, single-sample functions and the filter takes care of batching
automatically.

Two Jacobian paths are supported:

- **Autodiff** (default): Jacobians are computed by `torch.func.jacrev`.
  Works for any differentiable `f`/`h` with no extra code from the user.
- **Analytic**: user provides `F_fn` and `H_fn` kwargs.  Faster and
  often more numerically stable for systems with known closed-form
  Jacobians.

---

## Class reference

```python
class ExtendedKalmanFilter:
    def __init__(
        self,
        f:    Callable[[Tensor, Tensor | None], Tensor],
        h:    Callable[[Tensor], Tensor],
        dim_x: int,
        dim_z: int,
        F_fn: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        H_fn: Callable[[Tensor], Tensor] | None = None,
    ) -> None
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `f` | `Callable` | — | State transition, unbatched: `(dim_x,) → (dim_x,)`. Second argument is the control input `u` or `None`. |
| `h` | `Callable` | — | Measurement function, unbatched: `(dim_x,) → (dim_z,)`. |
| `dim_x` | `int` | — | State dimension. |
| `dim_z` | `int` | — | Measurement dimension. |
| `F_fn` | `Callable` or `None` | `None` | Optional analytic Jacobian of `f`: `(dim_x,) → (dim_x, dim_x)`. If omitted, `jacrev` is used. |
| `H_fn` | `Callable` or `None` | `None` | Optional analytic Jacobian of `h`: `(dim_x,) → (dim_z, dim_x)`. If omitted, `jacrev` is used. |

---

### `predict(x, P, Q, u=None) → (x_pred, P_pred)`

EKF predict: propagate the mean through the nonlinear `f`, propagate
the covariance through the Jacobian `F = ∂f/∂x` evaluated at `x`.

```
x_pred = f(x)                 # nonlinear mean propagation
F      = ∂f/∂x  at x
P_pred = F @ P @ F.T + Q      # linearised covariance propagation
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `x` | `(B, dim_x)` | Prior state estimates |
| `P` | `(B, dim_x, dim_x)` | Prior covariances |
| `Q` | `(dim_x, dim_x)` | Process noise covariance |
| `u` | `(B, dim_u)` or `None` | Optional control inputs |

Returns `(x_pred, P_pred)`.

---

### `update(x, P, z, R) → (x_post, P_post, y, S)`

EKF update: linearise `h` at the predicted state, then apply the
standard Kalman update with Joseph-form covariance.

```
z_pred = h(x)             # nonlinear measurement prediction
y      = z − z_pred       # innovation
H      = ∂h/∂x  at x
S      = H @ P @ H.T + R
K      = P @ H.T @ S⁻¹   (via cholesky_solve)
x_post = x + K @ y
P_post = (I − K H) P (I − K H).T + K R K.T  (Joseph form)
```

Returns `(x_post, P_post, y, S)`.

---

## Usage

### Autodiff Jacobians (default)

```python
import torch
from automatic_kalman.filters.ekf import ExtendedKalmanFilter

# Duffing oscillator (discrete-time Euler integration)
DT, K, K3, C = 0.05, 1.0, 0.5, 0.2

def f(x: torch.Tensor, u) -> torch.Tensor:
    pos, vel = x[0], x[1]
    acc = -K * pos - K3 * pos**3 - C * vel
    return torch.stack([pos + DT * vel, vel + DT * acc])

def h(x: torch.Tensor) -> torch.Tensor:
    return x[:1]  # observe position only

ekf = ExtendedKalmanFilter(f=f, h=h, dim_x=2, dim_z=1)
```

### Analytic Jacobians (opt-in)

```python
def F_fn(x: torch.Tensor, u) -> torch.Tensor:
    pos, vel = x[0], x[1]
    return torch.tensor([
        [1.0,                               DT],
        [-DT * (K + 3 * K3 * pos**2), 1.0 - DT * C],
    ], dtype=x.dtype)

def H_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0]], dtype=x.dtype)

ekf = ExtendedKalmanFilter(f=f, h=h, dim_x=2, dim_z=1, F_fn=F_fn, H_fn=H_fn)
```

### Training

The EKF drops into `train()` unchanged:

```python
from automatic_kalman.train import train
from automatic_kalman.losses import nis_consistency
from automatic_kalman.covariance import CholeskyCovariance

cov_Q = CholeskyCovariance(dim=2)
cov_R = CholeskyCovariance(dim=1)

losses = train(
    filter=ekf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nis_consistency,
    loader=loader,
    x0=torch.zeros(2, dtype=torch.float64),
    P0=torch.eye(2, dtype=torch.float64),
    epochs=150, lr=0.03,
)
```

## Notes

- `f` and `h` must accept and return `float64` tensors.
- `f` takes two arguments: state `x` and control `u` (or `None`).
  `h` takes only the state.
- When `jacrev` is used, every operation inside `f` and `h` must be
  differentiable.  This is also a requirement for `gradcheck` to pass.
- On linear `f`/`h`, the EKF is numerically equivalent to the KF.
  The test suite verifies this property.
