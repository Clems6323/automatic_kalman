# CholeskyCovariance

```python
from automatic_kalman.covariance import CholeskyCovariance
```

## Overview

`CholeskyCovariance` is the sole learnable module in the framework.
It wraps a lower-triangular Cholesky factor `L` as an `nn.Parameter`
and reconstructs the full covariance matrix on demand:

```
cov = L @ L.T + eps * I
```

**Why not optimize Q/R directly?**  Covariance matrices must stay
symmetric positive-definite throughout training.  The Cholesky
parameterisation enforces this constraint structurally — any value of
the raw parameter `_L_raw` produces a valid SPD matrix.

**Diagonal positivity.**  The diagonal entries of `L` are passed through
`softplus` before use, enforcing strict positivity and avoiding the
sign ambiguity of raw Cholesky.  Off-diagonal entries are unconstrained.

**Default initialisation.**  `factor()` returns the identity matrix, so
`matrix()` returns `(1 + eps) * I` at construction time.  This is a
reasonable starting point — scale adjusts quickly under gradient descent.

---

## Class reference

```python
class CholeskyCovariance(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dim` | `int` | — | Matrix dimension. |
| `eps` | `float` | `1e-6` | Jitter constant added to the diagonal. Not tunable — keep it small and fixed. |

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `dim` | `int` | Matrix dimension (as passed to `__init__`). |
| `eps` | `float` | Jitter constant (as passed to `__init__`). |
| `_L_raw` | `nn.Parameter` | Raw lower-triangular parameter, shape `(dim, dim)`. The diagonal is interpreted through `softplus`; entries above the diagonal are zero but not constrained. |

### Methods

#### `factor() → Tensor`

Return the lower-triangular Cholesky factor `L`, shape `(dim, dim)`.

- The diagonal is `softplus(_L_raw.diagonal())` — strictly positive.
- Off-diagonals are the raw lower-triangular entries — unconstrained.
- Always in the autograd graph.

#### `matrix() → Tensor`

Return the full SPD covariance matrix `L @ L.T + eps * I`,
shape `(dim, dim)`.

- Always SPD.
- Always in the autograd graph.
- This is the tensor to pass as `Q` or `R` to a filter.

---

## Usage

```python
import torch
from automatic_kalman.covariance import CholeskyCovariance

# Process noise for a 2-D state (position, velocity)
cov_Q = CholeskyCovariance(dim=2)

# Measurement noise for a 1-D observation
cov_R = CholeskyCovariance(dim=1)

# Retrieve tensors for the filter
Q = cov_Q.matrix()   # (2, 2) — in the autograd graph
R = cov_R.matrix()   # (1, 1)

# Register parameters with an optimizer
optimizer = torch.optim.Adam(
    list(cov_Q.parameters()) + list(cov_R.parameters()),
    lr=1e-2,
)
```

## Moving to GPU

```python
cov_Q = CholeskyCovariance(dim=2).cuda()
cov_R = CholeskyCovariance(dim=1).cuda()
# train() infers the device from cov_Q automatically
```

## Notes

- Never call `torch.linalg.cholesky` on the output of `matrix()` at
  runtime — you already have the factor via `factor()`.
- Do not add the `eps` jitter to the optimizer parameter list; it is a
  constant, not a learnable parameter.
- `_L_raw` is the only `nn.Parameter`.  Iterating `cov.parameters()`
  yields exactly one tensor.
