# KalmanFilter

```python
from automatic_kalman.filters.kf import KalmanFilter
```

## Overview

The linear Kalman filter.  This is the reference implementation — all
other filters in this package are validated by reducing to it on
linear-Gaussian data.

Numerical design choices enforced here:

- Kalman gain computed via `torch.cholesky_solve`, never `torch.inverse`.
- Covariance update uses the **Joseph form** for numerical stability and
  gradient conditioning.
- Covariances symmetrised with `0.5 * (P + P.T)` after every update.
- All computations in `float64`.

---

## Class reference

```python
class KalmanFilter:
    def __init__(self, F: Tensor, H: Tensor) -> None
```

### Parameters

| Name | Shape | Description |
|------|-------|-------------|
| `F` | `(dim_x, dim_x)` | State transition matrix. Cast to `float64` on construction. |
| `H` | `(dim_z, dim_x)` | Measurement matrix. Cast to `float64` on construction. |

`F` and `H` are part of the **system model**, not learned parameters.
They are fixed at construction time and moved to the input device/dtype
at runtime so the filter works on any device.

---

### `predict(x, P, Q, u=None) → (x_pred, P_pred)`

Linear prediction step:

```
x_pred = F @ x
P_pred = F @ P @ F.T + Q
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `x` | `(B, dim_x)` | Prior state estimates |
| `P` | `(B, dim_x, dim_x)` | Prior covariances |
| `Q` | `(dim_x, dim_x)` | Process noise covariance (from `CholeskyCovariance.matrix()`) |
| `u` | any or `None` | Ignored — no control input in the linear model |

Returns `(x_pred, P_pred)` with shapes `(B, dim_x)` and `(B, dim_x, dim_x)`.

---

### `update(x, P, z, R) → (x_post, P_post, y, S)`

Linear update step using Cholesky-solve for the Kalman gain and
Joseph form for the covariance:

```
y     = z − H @ x_pred                         # innovation
S     = H @ P @ H.T + R                        # innovation covariance
K     = P @ H.T @ S⁻¹  (via cholesky_solve)
x_post = x + K @ y
P_post = (I − K H) P (I − K H).T + K R K.T   # Joseph form
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `x` | `(B, dim_x)` | Predicted state estimates |
| `P` | `(B, dim_x, dim_x)` | Predicted covariances |
| `z` | `(B, dim_z)` | Measurements |
| `R` | `(dim_z, dim_z)` | Measurement noise covariance |

Returns `(x_post, P_post, y, S)`:

| Return | Shape |
|--------|-------|
| `x_post` | `(B, dim_x)` |
| `P_post` | `(B, dim_x, dim_x)` |
| `y` | `(B, dim_z)` |
| `S` | `(B, dim_z, dim_z)` |

---

## Usage

```python
import torch
from automatic_kalman.filters.kf import KalmanFilter
from automatic_kalman.covariance import CholeskyCovariance

dt = 0.1
F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64)
H = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

kf    = KalmanFilter(F, H)
cov_Q = CholeskyCovariance(dim=2)
cov_R = CholeskyCovariance(dim=1)

# One manual step
B = 32
x = torch.zeros(B, 2, dtype=torch.float64)
P = torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).clone()
z = torch.randn(B, 1, dtype=torch.float64)

Q = cov_Q.matrix()
R = cov_R.matrix()

x, P        = kf.predict(x, P, Q)
x, P, y, S  = kf.update(x, P, z, R)
```

## End-to-end training

Pass the `KalmanFilter` instance directly to [`train()`](train.md):

```python
from automatic_kalman.train import train
from automatic_kalman.losses import nis_consistency
from torch.utils.data import DataLoader

losses = train(
    filter=kf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nis_consistency,
    loader=loader,
    x0=torch.zeros(2, dtype=torch.float64),
    P0=torch.eye(2, dtype=torch.float64),
    epochs=200, lr=0.02,
)
```

## Validation

The implementation is tested against a NumPy baseline:
on linear-Gaussian data the KF must match the reference to `1e-10` in `float64`.
See `tests/test_against_baseline.py`.
