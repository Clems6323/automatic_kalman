# Losses

```python
from automatic_kalman.losses import nis_consistency, nees_consistency
```

## Overview

Two consistency losses are provided, both using the same
`(FilterOutputs, ...) → scalar Tensor` signature so the training loop
is loss-agnostic.

Both losses follow the same structure:

```
loss = (mean_statistic - expected_value)² + λ · mean(log det covariance)
```

The log-det regulariser prevents the trivial minimiser of inflating `R`
(NIS) or `P` (NEES) to drive the Mahalanobis metric to zero.  Without
it, the optimiser learns to over-inflate covariances and the statistic
converges to 0 rather than `dim_z`/`dim_x`.

---

## `nis_consistency`

```python
def nis_consistency(outputs: FilterOutputs, lam: float = 1.0) -> Tensor
```

**NIS = Normalised Innovation Squared.**  Use this loss when
ground-truth states are **not available**.

Under a correctly tuned filter the expected NIS at each step is
`dim_z` (the measurement dimension).  The loss minimises the squared
deviation from this expected value:

```
NIS_k  = yₖᵀ Sₖ⁻¹ yₖ           (computed via cholesky_solve — no matrix inverse)
loss   = (mean(NIS) − dim_z)² + λ · mean(log det S)
```

### Arguments

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `outputs` | `FilterOutputs` | — | Filter outputs for one full T-step pass. Must contain `innovations` and `innovation_covs`. |
| `lam` | `float` | `1.0` | Weight on the log-det regulariser. |

### Returns

Scalar `Tensor` with gradient to `L_Q` and `L_R`.

### Example

```python
from automatic_kalman.losses import nis_consistency

losses = train(
    filter=kf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nis_consistency,       # no ground truth needed
    loader=loader,
    x0=x0, P0=P0,
)
```

To pass a non-default `lam`:

```python
import functools
loss_fn = functools.partial(nis_consistency, lam=0.5)
```

---

## `nees_consistency`

```python
def nees_consistency(outputs: FilterOutputs, lam: float = 1.0) -> Tensor
```

**NEES = Normalised Estimation Error Squared.**  Use this loss when
ground-truth states **are available**.

Under a correctly tuned filter the expected NEES at each step is
`dim_x` (the state dimension):

```
eₖ     = xₖ − xₖ*              (estimation error)
NEES_k = eₖᵀ Pₖ⁻¹ eₖ          (computed via cholesky_solve)
loss   = (mean(NEES) − dim_x)² + λ · mean(log det P)
```

### Arguments

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `outputs` | `FilterOutputs` | — | Filter outputs. Must contain `states_true` (not `None`). |
| `lam` | `float` | `1.0` | Weight on the log-det regulariser. |

### Returns

Scalar `Tensor` with gradient to `L_Q` and `L_R`.

### Raises

`ValueError` — if `outputs.states_true` is `None`.

### Example

```python
from automatic_kalman.losses import nees_consistency

losses = train(
    filter=kf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nees_consistency,      # requires ground truth in loader
    loader=loader,
    x0=x0, P0=P0,
)
```

---

## Choosing between NIS and NEES

| | NIS | NEES |
|--|-----|------|
| Requires ground truth | No | Yes |
| Optimises | Measurement noise `R` (primarily) | State noise `Q` (primarily) |
| Expected value | `dim_z` | `dim_x` |
| Regularises | `log det S` | `log det P` |

In practice, either loss recovers both Q and R — the cross-coupling
comes from the filter equations.  NIS is the more common choice for
real applications where ground truth is unavailable.

---

## Writing a custom loss

Custom losses must follow the `(FilterOutputs, ...) → scalar Tensor`
signature:

```python
def my_loss(outputs: FilterOutputs, lam: float = 1.0) -> torch.Tensor:
    y = outputs.innovations        # (B, T, dim_z)
    S = outputs.innovation_covs    # (B, T, dim_z, dim_z)
    ...
    return scalar_tensor
```

The returned tensor must preserve the autograd graph connected to
`L_Q` and `L_R`.  Never call `.detach()` or `.item()` on intermediate
values that flow into the output.
