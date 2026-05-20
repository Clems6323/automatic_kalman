# FilterOutputs

```python
from automatic_kalman.types import FilterOutputs
```

## Overview

`FilterOutputs` is a dataclass that carries all per-step tensors
collected by the training loop over one full T-step pass through a
batch.  It is the sole input consumed by the loss functions.

All tensors in `FilterOutputs` remain in the autograd graph.  Do not
call `.detach()` on any field before passing the object to a loss.

---

## Reference

```python
@dataclass
class FilterOutputs:
    innovations:     Tensor          # (B, T, dim_z)
    innovation_covs: Tensor          # (B, T, dim_z, dim_z)
    states:          Tensor          # (B, T, dim_x)
    state_covs:      Tensor          # (B, T, dim_x, dim_x)
    states_true:     Tensor | None   # (B, T, dim_x) or None
```

### Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `innovations` | `(B, T, dim_z)` | Per-step innovations `y = z − h(x_pred)` from `filter.update`. |
| `innovation_covs` | `(B, T, dim_z, dim_z)` | Per-step innovation covariances `S` from `filter.update`. |
| `states` | `(B, T, dim_x)` | Per-step posterior state estimates from `filter.update`. |
| `state_covs` | `(B, T, dim_x, dim_x)` | Per-step posterior covariances from `filter.update`. |
| `states_true` | `(B, T, dim_x)` or `None` | Ground-truth states when available (from the data loader). `None` when using `nis_consistency`. |

---

## How it is constructed

`train()` builds a `FilterOutputs` at the end of each batch's T-step
inner loop by stacking the per-step outputs:

```python
outputs = FilterOutputs(
    innovations=torch.stack(innovations_list, dim=1),         # (B, T, dim_z)
    innovation_covs=torch.stack(innovation_covs_list, dim=1), # (B, T, dim_z, dim_z)
    states=torch.stack(states_list, dim=1),                   # (B, T, dim_x)
    state_covs=torch.stack(state_covs_list, dim=1),           # (B, T, dim_x, dim_x)
    states_true=x_true_batch,                                  # None or (B, T, dim_x)
)
```

---

## Writing a custom loss

Custom losses receive `FilterOutputs` and should return a scalar
`Tensor` in the autograd graph:

```python
from automatic_kalman.types import FilterOutputs
import torch

def log_likelihood_loss(outputs: FilterOutputs, lam: float = 1.0) -> torch.Tensor:
    y = outputs.innovations        # (B, T, dim_z)
    S = outputs.innovation_covs    # (B, T, dim_z, dim_z)

    L_S = torch.linalg.cholesky(S)
    log_det_S = 2.0 * L_S.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (B, T)
    S_inv_y = torch.cholesky_solve(y.unsqueeze(-1), L_S).squeeze(-1)
    nll = 0.5 * ((y * S_inv_y).sum(-1) + log_det_S)

    return nll.mean()
```

Pass it directly to `train()`:

```python
from automatic_kalman.train import train
epoch_losses = train(..., loss_fn=log_likelihood_loss)
```
