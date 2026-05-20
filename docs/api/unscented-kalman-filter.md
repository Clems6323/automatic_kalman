# UnscentedKalmanFilter

```python
from automatic_kalman.filters.ukf import UnscentedKalmanFilter
```

## Overview

The Unscented Kalman filter using Merwe scaled sigma points.  Unlike the
EKF, the UKF propagates a deterministic set of 2n+1 carefully chosen
sigma points through the nonlinear functions `f` and `h` without
computing Jacobians.  This gives a third-order accurate mean and
second-order accurate covariance for Gaussian distributions.

Key properties of this implementation:

- Sigma-point generation uses `torch.linalg.cholesky(P)`, which is
  differentiable in modern PyTorch — gradients flow through to `L_Q`
  and `L_R`.
- `alpha`, `beta`, `kappa` are **fixed** during Q/R training.  Do not
  include them in the optimizer.
- For linear `f` and `h`, the UKF is algebraically equivalent to the
  classic KF (verified in the test suite).
- The `update` step regenerates sigma points from the predicted `(x, P)`.
  This is the standard two-step UKF formulation required to satisfy the
  `predict` / `update` Filter protocol.

---

## Class reference

```python
class UnscentedKalmanFilter:
    def __init__(
        self,
        f:     Callable[[Tensor, Tensor | None], Tensor],
        h:     Callable[[Tensor], Tensor],
        dim_x: int,
        dim_z: int,
        alpha: float = 1e-3,
        beta:  float = 2.0,
        kappa: float = 0.0,
    ) -> None
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `f` | `Callable` | — | State transition, unbatched: `(dim_x,) → (dim_x,)`. Second argument is control `u` or `None`. |
| `h` | `Callable` | — | Measurement function, unbatched: `(dim_x,) → (dim_z,)`. |
| `dim_x` | `int` | — | State dimension. |
| `dim_z` | `int` | — | Measurement dimension. |
| `alpha` | `float` | `1e-3` | Controls the spread of sigma points around the mean. Typical range: `[1e-3, 1]`. |
| `beta` | `float` | `2.0` | Encodes prior knowledge of the distribution. `2` is optimal for Gaussian. |
| `kappa` | `float` | `0.0` | Secondary scaling parameter. |

!!! warning "Alpha for gradient-based tests"
    The default `alpha=1e-3` produces `W_m[0] ≈ −999 999`, which causes
    catastrophic floating-point cancellation on linear systems.  For
    gradcheck and reduces-to-KF tests use `alpha=0.3`.  For statistical
    consistency tests on real nonlinear data the default is fine.

---

## Sigma-point weights

The Merwe weight vectors `W_m` (mean) and `W_c` (covariance) have
2n+1 entries.  With `n = dim_x` and `λ = α² (n + κ) − n`:

```
W_m[0]    = λ / (n + λ)
W_m[i]    = 1 / (2 (n + λ))   for i = 1..2n
W_c[0]    = λ / (n + λ) + (1 − α² + β)   # β correction
W_c[i]    = W_m[i]             for i = 1..2n
```

`W_m` sums exactly to 1.  `W_c[0] ≠ W_m[0]` (the β correction shifts
the zeroth weight to encode higher-order distribution information).

---

### `predict(x, P, Q, u=None) → (x_pred, P_pred)`

1. Generate 2n+1 sigma points from `(x, P)` via `cholesky(P)`.
2. Propagate each through `f`.
3. Reconstruct weighted mean and covariance, add `Q`.

```
x_pred = Σ_i W_m[i] · f(X_i)
P_pred = Σ_i W_c[i] · (f(X_i) − x_pred)(f(X_i) − x_pred).T + Q
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

1. Regenerate sigma points from the predicted `(x, P)`.
2. Transform each through `h`.
3. Compute weighted measurement mean, innovation covariance `S`, and
   cross-covariance `P_xz`.
4. Kalman gain via `cholesky_solve(P_xz.T, L_S)`.
5. State and covariance update.

```
z_pred = Σ_i W_m[i] · h(X_i)
y      = z − z_pred
S      = Σ_i W_c[i] · δz_i δz_i.T + R
P_xz   = Σ_i W_c[i] · δx_i δz_i.T
K      = P_xz @ S⁻¹              (via cholesky_solve)
x_post = x + K @ y
P_post = P − K @ S @ K.T
```

Returns `(x_post, P_post, y, S)`.

---

## Usage

```python
import torch
from automatic_kalman.filters.ukf import UnscentedKalmanFilter

DT, K, K3, C = 0.05, 1.0, 0.5, 0.2

def f(x: torch.Tensor, u) -> torch.Tensor:
    pos, vel = x[0], x[1]
    acc = -K * pos - K3 * pos**3 - C * vel
    return torch.stack([pos + DT * vel, vel + DT * acc])

def h(x: torch.Tensor) -> torch.Tensor:
    return x[:1]

ukf = UnscentedKalmanFilter(f=f, h=h, dim_x=2, dim_z=1)
```

### Training

```python
from automatic_kalman.train import train
from automatic_kalman.losses import nis_consistency
from automatic_kalman.covariance import CholeskyCovariance

cov_Q = CholeskyCovariance(dim=2)
cov_R = CholeskyCovariance(dim=1)

losses = train(
    filter=ukf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nis_consistency,
    loader=loader,
    x0=torch.zeros(2, dtype=torch.float64),
    P0=torch.eye(2, dtype=torch.float64),
    epochs=150, lr=0.03,
)
```

## Notes

- `f` and `h` are unbatched callables — same signature as for
  [`ExtendedKalmanFilter`](extended-kalman-filter.md).  The UKF handles
  batching internally via `torch.func.vmap`.
- A failed `cholesky(P)` triggers one retry with a `1e-8` jitter.
  If that also fails, a `RuntimeError` is raised.  Silent fallbacks are
  not provided because they mask real divergence.
- Do not replace `cholesky` with SVD unless you have a concrete failure.
  SVD gradients are more expensive and less stable at repeated singular
  values.
