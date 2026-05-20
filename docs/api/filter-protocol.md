# Filter Protocol

```python
from automatic_kalman.filters.base import Filter
```

## Overview

`Filter` is a `typing.Protocol` that defines the interface all
differentiable filters must implement.  The training loop, losses, and
tests never inspect the concrete filter type — they interact exclusively
through this protocol.

Implementing the protocol means providing two methods with exact
signatures: `predict` and `update`.

---

## Protocol reference

```python
class Filter(Protocol):
    def predict(
        self,
        x: Tensor,
        P: Tensor,
        Q: Tensor,
        u: Tensor | None,
    ) -> tuple[Tensor, Tensor]: ...

    def update(
        self,
        x: Tensor,
        P: Tensor,
        z: Tensor,
        R: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
```

### Tensor shapes

All methods operate on **batched** inputs.  `B` is the batch dimension
(number of independent trajectories processed in parallel).

| Symbol | Shape | Description |
|--------|-------|-------------|
| `x` | `(B, dim_x)` | Batch of state estimates |
| `P` | `(B, dim_x, dim_x)` | Batch of covariance matrices |
| `Q` | `(dim_x, dim_x)` | Shared process noise covariance (single matrix, not batched) |
| `R` | `(dim_z, dim_z)` | Shared measurement noise covariance (single matrix, not batched) |
| `z` | `(B, dim_z)` | Batch of measurements at one time step |
| `u` | `(B, dim_u)` or `None` | Optional control inputs |

---

### `predict(x, P, Q, u) → (x_pred, P_pred)`

Propagate the state and covariance forward one time step.

**Arguments**

| Name | Shape | Description |
|------|-------|-------------|
| `x` | `(B, dim_x)` | Prior state estimates |
| `P` | `(B, dim_x, dim_x)` | Prior covariances |
| `Q` | `(dim_x, dim_x)` | Process noise covariance |
| `u` | `(B, dim_u)` or `None` | Optional control inputs |

**Returns**

| Name | Shape | Description |
|------|-------|-------------|
| `x_pred` | `(B, dim_x)` | Predicted state estimates |
| `P_pred` | `(B, dim_x, dim_x)` | Predicted covariances |

---

### `update(x, P, z, R) → (x_post, P_post, y, S)`

Incorporate a batch of measurements.

**Arguments**

| Name | Shape | Description |
|------|-------|-------------|
| `x` | `(B, dim_x)` | Predicted states (output of `predict`) |
| `P` | `(B, dim_x, dim_x)` | Predicted covariances (output of `predict`) |
| `z` | `(B, dim_z)` | Measurements at the current time step |
| `R` | `(dim_z, dim_z)` | Measurement noise covariance |

**Returns**

| Name | Shape | Description |
|------|-------|-------------|
| `x_post` | `(B, dim_x)` | Posterior state estimates |
| `P_post` | `(B, dim_x, dim_x)` | Posterior covariances |
| `y` | `(B, dim_z)` | Innovations `z − h(x_pred)` |
| `S` | `(B, dim_z, dim_z)` | Innovation covariances |

!!! note "`y` and `S` must be returned"
    The loss functions [`nis_consistency`](losses.md#nis_consistency) and
    [`nees_consistency`](losses.md#nees_consistency) consume `y` and `S`
    from `FilterOutputs`.  They must remain in the autograd graph.
    Do not recompute them outside the filter.

---

## Concrete implementations

| Class | Module |
|-------|--------|
| [`KalmanFilter`](kalman-filter.md) | `automatic_kalman.filters.kf` |
| [`ExtendedKalmanFilter`](extended-kalman-filter.md) | `automatic_kalman.filters.ekf` |
| [`UnscentedKalmanFilter`](unscented-kalman-filter.md) | `automatic_kalman.filters.ukf` |

---

## Implementing a new filter

A new filter variant (e.g. square-root UKF, IEKF) only needs to
implement `predict` and `update` with the signatures above.  It drops
into `train()` and all consistency tests without any other changes.

If you find yourself adding a branch inside `train()` or a loss function
for a new filter type, that is a sign the protocol is being violated.
