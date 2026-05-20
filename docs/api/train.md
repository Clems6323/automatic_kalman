# train()

```python
from automatic_kalman.train import train
```

## Overview

The filter-agnostic training loop.  It accepts any object satisfying the
[`Filter` protocol](filter-protocol.md) and optimises the Cholesky
factors of `Q` and `R` using Adam.

Key features:

- Works with KF, EKF, UKF, or any custom filter — no branches on type.
- Gradient clipping on `L_Q` and `L_R` (norm ≤ `grad_clip`) to handle
  gradient spikes near singular `S`.
- Condition-number monitoring: DEBUG log at each epoch; WARNING when
  `cond(S)` or `cond(P)` exceeds `cond_warn_threshold`.
- Device inference from `cov_Q` parameters — move `cov_Q.cuda()` before
  calling to enable GPU training; all inputs are cast automatically.
- Supports loaders that yield `(z,)` or `(z, x_true)` tuples.

---

## Function reference

```python
def train(
    filter: Filter,
    cov_Q: CholeskyCovariance,
    cov_R: CholeskyCovariance,
    loss_fn: Callable[[FilterOutputs], Tensor],
    loader: DataLoader,
    *,
    x0: Tensor,
    P0: Tensor,
    epochs: int = 100,
    lr: float = 1e-2,
    grad_clip: float = 10.0,
    cond_warn_threshold: float = 1e8,
) -> list[float]
```

### Arguments

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `filter` | `Filter` | — | Any object implementing `predict` / `update`. |
| `cov_Q` | `CholeskyCovariance` | — | Learnable process noise covariance. |
| `cov_R` | `CholeskyCovariance` | — | Learnable measurement noise covariance. |
| `loss_fn` | `Callable[[FilterOutputs], Tensor]` | — | Consistency loss, e.g. `nis_consistency` or `nees_consistency`. |
| `loader` | `DataLoader` | — | Yields `(z_batch,)` or `(z_batch, x_true_batch)`. Tensors are cast to `float64` and moved to device internally. |
| `x0` | `Tensor`, shape `(dim_x,)` | — | Initial state estimate. |
| `P0` | `Tensor`, shape `(dim_x, dim_x)` | — | Initial covariance. |
| `epochs` | `int` | `100` | Number of full passes over the dataset. |
| `lr` | `float` | `1e-2` | Adam learning rate. |
| `grad_clip` | `float` | `10.0` | Maximum gradient norm applied to `L_Q` and `L_R`. |
| `cond_warn_threshold` | `float` | `1e8` | Emit WARNING when `cond(S)` or `cond(P)` exceeds this. |

### Returns

`list[float]` — per-epoch average loss as Python floats.  Plot this to
monitor convergence.

---

## Training loop internals

For each epoch, for each batch `(z_batch, x_true_batch)`:

1. Build `Q = cov_Q.matrix()`, `R = cov_R.matrix()` — both in the
   autograd graph.
2. Broadcast `x0`, `P0` across the batch dimension.
3. Run a T-step inner loop: `predict` then `update` at each time step.
4. Collect all `(y, S, x_post, P_post)` into a `FilterOutputs` tensor.
5. Call `loss_fn(outputs)`, run `backward()`.
6. Clip gradients on `L_Q`, `L_R`; step optimizer.
7. After the last batch in the epoch, log `cond(S)` and `cond(P)`.

---

## Usage

### NIS loss (no ground truth)

```python
import torch
from torch.utils.data import DataLoader
from automatic_kalman.filters.kf import KalmanFilter
from automatic_kalman.covariance import CholeskyCovariance
from automatic_kalman.losses import nis_consistency
from automatic_kalman.train import train

kf    = KalmanFilter(F, H)
cov_Q = CholeskyCovariance(dim=2)
cov_R = CholeskyCovariance(dim=1)

# Loader yielding (z_batch,) — no ground truth
loader = DataLoader(dataset, batch_size=20, shuffle=True)

epoch_losses = train(
    filter=kf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nis_consistency,
    loader=loader,
    x0=torch.zeros(2, dtype=torch.float64),
    P0=torch.eye(2, dtype=torch.float64),
    epochs=200, lr=0.02,
)
```

### NEES loss (with ground truth)

```python
from automatic_kalman.losses import nees_consistency

# Loader yielding (z_batch, x_true_batch) tuples
loader = DataLoader(dataset_with_truth, batch_size=20, shuffle=True)

epoch_losses = train(
    filter=kf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nees_consistency,
    loader=loader,
    x0=torch.zeros(2, dtype=torch.float64),
    P0=torch.eye(2, dtype=torch.float64),
    epochs=200, lr=0.02,
)
```

### GPU training

```python
device = torch.device("cuda")
cov_Q = CholeskyCovariance(dim=2).to(device)
cov_R = CholeskyCovariance(dim=1).to(device)

# train() infers the device from cov_Q — no other change needed
epoch_losses = train(
    filter=kf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nis_consistency,
    loader=loader,
    x0=torch.zeros(2, dtype=torch.float64),
    P0=torch.eye(2, dtype=torch.float64),
)
```

### Condition-number diagnostics

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# DEBUG output now shows cond(S) and cond(P) at each epoch.
# To lower the warning threshold:
epoch_losses = train(
    ...,
    cond_warn_threshold=1e6,   # warn earlier
)
```

## Notes

- `train()` itself is unchanged by which filter type is used.  Swapping
  `KalmanFilter` for `ExtendedKalmanFilter` or `UnscentedKalmanFilter`
  requires only changing the `filter=` argument.
- The optimizer is Adam; the parameters are the Cholesky factors of
  `cov_Q` and `cov_R`.  No other parameters are tuned.
- The condition-number computation is wrapped in `torch.no_grad()` and
  uses `.detach()` — it never touches the autograd graph.
