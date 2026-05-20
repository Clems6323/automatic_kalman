# API Reference

`automatic_kalman` is organised into a small set of focused modules.
Every public symbol is documented on its own page below.

---

## Modules

| Module | Symbols |
|--------|---------|
| [`automatic_kalman.covariance`](covariance.md) | [`CholeskyCovariance`](covariance.md) |
| [`automatic_kalman.filters.base`](filter-protocol.md) | [`Filter`](filter-protocol.md) |
| [`automatic_kalman.filters.kf`](kalman-filter.md) | [`KalmanFilter`](kalman-filter.md) |
| [`automatic_kalman.filters.ekf`](extended-kalman-filter.md) | [`ExtendedKalmanFilter`](extended-kalman-filter.md) |
| [`automatic_kalman.filters.ukf`](unscented-kalman-filter.md) | [`UnscentedKalmanFilter`](unscented-kalman-filter.md) |
| [`automatic_kalman.losses`](losses.md) | [`nis_consistency`](losses.md#nis_consistency), [`nees_consistency`](losses.md#nees_consistency) |
| [`automatic_kalman.train`](train.md) | [`train`](train.md) |
| [`automatic_kalman.types`](types.md) | [`FilterOutputs`](types.md) |
| [`automatic_kalman.data.generators`](generators.md) | [`TrajectoryBatch`](generators.md#trajectorybatch), [`LinearGaussianGenerator`](generators.md#lineargaussiangenerator), [`NonlinearGenerator`](generators.md#nonlineargenerator), [`TrajectoryDataset`](generators.md#trajectorydataset) |

---

## Design invariants

These invariants hold across every module and must be preserved when
adding new filters, losses, or generators.

**Float64 everywhere.**  All filter computations run in `float64`.
`float32` introduces NIS/NEES bias that is indistinguishable from a
tuning problem but is actually roundoff.  Mixed-precision inputs are cast
at the training-loop boundary.

**Autograd graph is never broken.**  No `.detach()`, `.item()`, or
`torch.no_grad()` inside any path that connects `(L_Q, L_R)` to the
loss.  Diagnostics (condition numbers) are wrapped in `with torch.no_grad():` 
and their results are never fed back into the graph.

**No explicit matrix inverse.**  `torch.inverse` is never called on `S`
or `P`.  The Kalman gain is computed via `torch.cholesky_solve`.

**Joseph-form covariance update.**  Used in KF and EKF to keep `P`
symmetric and positive-semidefinite under gradient flow.

**Filter protocol is uniform.**  All filters implement `predict` /
`update` with identical signatures.  The training loop, losses, and
generators contain no filter-type branches.
