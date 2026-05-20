# Linear KF Tracking

**Notebook:** `notebooks/01_linear_kf_tracking.ipynb`

This notebook demonstrates the full covariance-learning pipeline on a
2D constant-velocity tracking problem — the simplest non-trivial Kalman
filter use case.  It uses the linear `KalmanFilter` and trains with both
the NIS loss (no ground truth) and the NEES loss (with ground truth).

---

## System

A target moves in the 2D plane with (approximately) constant velocity.
The state is `[x, y, vx, vy]` and only the position `[x, y]` is
observed:

```
F = block_diag([[1, dt], [0, 1]])   # dim_x = 4
H = [I₂ | 0]                        # dim_z = 2, observe x and y
```

True covariances used to generate the data:

```python
Q_true = diag([1e-4, 1e-4, 1e-2, 1e-2])   # small position noise, larger velocity
R_true = 0.5 * I₂                           # moderate measurement noise
```

Initial state: `x0 = [0, 0, 1, 0.5]` (start at origin, move NE).

---

## What the notebook covers

### Data generation

200 training trajectories of length 50 are generated with
`LinearGaussianGenerator`.  A separate held-out set of 500 trajectories
is used for evaluation.

### Training — NIS loss

`CholeskyCovariance` modules for Q (dim 4) and R (dim 2) are
initialised at `10×` the true values.  After 200 epochs of Adam
(`lr=0.02`) with the NIS consistency loss, the learned matrices
converge close to the true values.

The NIS loss requires no ground-truth states — only measurements are
used.

### Training — NEES loss

The same setup is repeated with the NEES consistency loss, which uses
ground-truth states from the loader.  NEES training is more direct
because it measures state-estimation error, but requires labelled data.

### Plots

**Loss curves** — both NIS and NEES losses decrease monotonically and
plateau near zero.

**Q/R recovery bar chart** — learned diagonal entries of Q and R are
shown alongside the true values.  Both losses recover the true values
to within ~5% after 200 epochs.

**NIS time series** — per-step NIS over the held-out evaluation set,
with the chi-squared 95% confidence band overlaid.  After training,
the NIS curve lies comfortably within the band, confirming the filter
is well-calibrated.

**2D track with 3σ ellipses** — the estimated trajectory is shown on
top of the true trajectory, with 3σ uncertainty ellipses drawn at
several timesteps.

**Comparison table:**

| | Q diag | R diag |
|--|--------|--------|
| True | `[1e-4, 1e-4, 1e-2, 1e-2]` | `[0.5, 0.5]` |
| NIS-learned | ~`[1.0e-4, ...]` | ~`[0.5, ...]` |
| NEES-learned | ~`[1.0e-4, ...]` | ~`[0.5, ...]` |
| Manual baseline (identity init) | diverges | — |

---

## Running the notebook

```bash
uv run jupyter lab notebooks/01_linear_kf_tracking.ipynb
```

The notebook runs top to bottom without errors and takes under 2 minutes
on CPU (PyTorch float64, batch size 20, 200 epochs).

---

## Key code fragments

### Filter and covariance setup

```python
from automatic_kalman.filters.kf import KalmanFilter
from automatic_kalman.covariance import CholeskyCovariance

dt = 0.1
F = torch.block_diag(
    torch.tensor([[1.0, dt], [0.0, 1.0]]),
    torch.tensor([[1.0, dt], [0.0, 1.0]]),
)
H = torch.zeros(2, 4, dtype=torch.float64)
H[0, 0] = H[1, 1] = 1.0

kf    = KalmanFilter(F, H)
cov_Q = CholeskyCovariance(dim=4)
cov_R = CholeskyCovariance(dim=2)
```

### NIS training

```python
from automatic_kalman.train import train
from automatic_kalman.losses import nis_consistency

losses_nis = train(
    filter=kf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nis_consistency,
    loader=train_loader,
    x0=x0, P0=P0,
    epochs=200, lr=0.02,
)
```
