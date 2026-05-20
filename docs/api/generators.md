# Data Generators

```python
from automatic_kalman.data.generators import (
    TrajectoryBatch,
    LinearGaussianGenerator,
    NonlinearGenerator,
    TrajectoryDataset,
)
```

## Overview

Three classes for generating synthetic trajectories used in training
and validation.  All outputs are `float64` and noise is sampled via
the Cholesky factors of the true covariances.

None of these classes are part of the autograd graph — they run under
`torch.no_grad()` and are used for data generation only.

---

## TrajectoryBatch

```python
@dataclass
class TrajectoryBatch:
    measurements: Tensor   # (B, T, dim_z)
    states:        Tensor   # (B, T, dim_x)
```

A plain dataclass returned by both generator classes.

| Field | Shape | Description |
|-------|-------|-------------|
| `measurements` | `(B, T, dim_z)` | Noisy observations. |
| `states` | `(B, T, dim_x)` | Ground-truth states. |

---

## LinearGaussianGenerator

```python
class LinearGaussianGenerator:
    def __init__(
        self,
        F: Tensor,      # (dim_x, dim_x)
        H: Tensor,      # (dim_z, dim_x)
        Q_true: Tensor, # (dim_x, dim_x)
        R_true: Tensor, # (dim_z, dim_z)
        x0: Tensor,     # (dim_x,)
    ) -> None

    def __call__(self, B: int, T: int) -> TrajectoryBatch
```

Generates batched trajectories from a **linear-Gaussian system**:

```
x_{k+1} = F x_k + w_k,    w_k ~ N(0, Q_true)
z_k     = H x_k + v_k,    v_k ~ N(0, R_true)
```

All inputs are cast to `float64` on construction.

### Parameters

| Name | Shape | Description |
|------|-------|-------------|
| `F` | `(dim_x, dim_x)` | State transition matrix. |
| `H` | `(dim_z, dim_x)` | Measurement matrix. |
| `Q_true` | `(dim_x, dim_x)` | True process noise covariance. Must be SPD. |
| `R_true` | `(dim_z, dim_z)` | True measurement noise covariance. Must be SPD. |
| `x0` | `(dim_x,)` | Initial state. |

### `__call__(B, T) → TrajectoryBatch`

Sample `B` independent trajectories of length `T`.

### Example

```python
import torch
from automatic_kalman.data.generators import LinearGaussianGenerator, TrajectoryDataset
from torch.utils.data import DataLoader

dt = 0.1
F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64)
H = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
Q_true = torch.diag(torch.tensor([1e-3, 1e-2], dtype=torch.float64))
R_true = torch.tensor([[0.5]], dtype=torch.float64)

gen    = LinearGaussianGenerator(F, H, Q_true, R_true, x0=torch.zeros(2))
batch  = gen(B=200, T=50)             # TrajectoryBatch

loader = DataLoader(
    TrajectoryDataset(batch),
    batch_size=20,
    shuffle=True,
)
```

---

## NonlinearGenerator

```python
class NonlinearGenerator:
    def __init__(
        self,
        f: Callable[[Tensor, None], Tensor],  # (dim_x,) → (dim_x,)
        h: Callable[[Tensor], Tensor],         # (dim_x,) → (dim_z,)
        Q_true: Tensor,
        R_true: Tensor,
        x0: Tensor,
    ) -> None

    def __call__(self, B: int, T: int) -> TrajectoryBatch
```

Generates batched trajectories from a **nonlinear dynamical system**:

```
x_{k+1} = f(x_k, None) + w_k,   w_k ~ N(0, Q_true)
z_k     = h(x_k) + v_k,          v_k ~ N(0, R_true)
```

`f` and `h` are **unbatched** callables.  Batching is handled internally
via `torch.func.vmap`.

### Example

```python
import torch
from automatic_kalman.data.generators import NonlinearGenerator, TrajectoryDataset
from torch.utils.data import DataLoader

DT, K, K3, C = 0.05, 1.0, 0.5, 0.2

def f(x, u):
    pos, vel = x[0], x[1]
    acc = -K * pos - K3 * pos**3 - C * vel
    return torch.stack([pos + DT * vel, vel + DT * acc])

def h(x):
    return x[:1]

Q_true = torch.diag(torch.tensor([1e-2, 1e-2], dtype=torch.float64))
R_true = torch.tensor([[0.1]], dtype=torch.float64)
x0     = torch.tensor([0.3, 0.0], dtype=torch.float64)

gen    = NonlinearGenerator(f=f, h=h, Q_true=Q_true, R_true=R_true, x0=x0)
batch  = gen(B=200, T=60)

loader = DataLoader(TrajectoryDataset(batch), batch_size=20, shuffle=True)
```

---

## TrajectoryDataset

```python
class TrajectoryDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, batch: TrajectoryBatch) -> None
```

A `torch.utils.data.Dataset` wrapping a `TrajectoryBatch`.
Each item is a `(measurements, states)` pair for a single trajectory.
Compatible with `DataLoader` out of the box.

`train()` expects the loader to yield `(z_batch, x_true_batch)` tuples,
which is exactly what `DataLoader(TrajectoryDataset(...))` produces.

When using `nis_consistency` (no ground truth required), it is safe to
pass this loader unchanged — `train()` reads both tensors but the
second is only forwarded to the loss when `states_true` is not `None`.
