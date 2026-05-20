# automatic-kalman

**Learn Kalman filter noise covariances Q and R by gradient descent.**

`automatic-kalman` trains the process-noise covariance **Q** and the measurement-noise
covariance **R** of a Bayesian filter entirely from data, using PyTorch autograd.
The same training loop works for all three filter families — linear (KF), Extended (EKF),
and Unscented (UKF) — through a uniform `predict` / `update` interface.

---

## Installation

```bash
git clone https://github.com/username/automatic-kalman
cd automatic-kalman
uv sync --extra dev   # installs torch, pytest, matplotlib, jupyter, mkdocs-material
uv run pytest         # 32 tests, all green
```

Requires **Python ≥ 3.12** and **PyTorch ≥ 2.5**.
GPU (CUDA 12.8) is included when the project-local `torch` wheel resolves to the CUDA build.

---

## Quick start

```python
import torch
from torch.utils.data import DataLoader

from automatic_kalman.covariance import CholeskyCovariance
from automatic_kalman.data.generators import LinearGaussianGenerator, TrajectoryDataset
from automatic_kalman.filters.kf import KalmanFilter
from automatic_kalman.losses import nis_consistency
from automatic_kalman.train import train

# ── 1. Define the system ─────────────────────────────────────────────────────
dt = 0.1
F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64)
H = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

Q_true = torch.diag(torch.tensor([1e-3, 1e-2], dtype=torch.float64))
R_true = torch.tensor([[0.5]], dtype=torch.float64)

# ── 2. Generate training data ────────────────────────────────────────────────
gen     = LinearGaussianGenerator(F, H, Q_true, R_true, x0=torch.zeros(2))
loader  = DataLoader(TrajectoryDataset(gen(B=200, T=50)), batch_size=20, shuffle=True)

# ── 3. Set up filter and learnable covariances ───────────────────────────────
kf    = KalmanFilter(F, H)
cov_Q = CholeskyCovariance(dim=2)   # parameter: lower-triangular factor of Q
cov_R = CholeskyCovariance(dim=1)   # parameter: lower-triangular factor of R

# ── 4. Train ─────────────────────────────────────────────────────────────────
losses = train(
    filter=kf, cov_Q=cov_Q, cov_R=cov_R,
    loss_fn=nis_consistency,          # no ground-truth state required
    loader=loader,
    x0=torch.zeros(2, dtype=torch.float64),
    P0=torch.eye(2, dtype=torch.float64),
    epochs=200, lr=0.02,
)

Q_learned = cov_Q.matrix()   # (2, 2)  SPD tensor, in the autograd graph
R_learned = cov_R.matrix()   # (1, 1)
```

Replacing `KalmanFilter` with `ExtendedKalmanFilter` or `UnscentedKalmanFilter` requires
**no change** to the `train` call or the loss function.

---

## Design principles

### Q and R are Cholesky-parameterised

`Q` and `R` are never optimised directly — they must remain symmetric positive-definite.
Instead, lower-triangular factors `L_Q`, `L_R` are learned and the full matrices are
reconstructed as `Q = L @ Lᵀ + ε I`.  The diagonal of `L` is mapped through `softplus`
to enforce strict positivity.  The map is smooth, avoids sign ambiguity, and the factor
is always in the autograd graph.

### Uniform Filter protocol

All three filters implement the same `predict(x, P, Q, u)` and `update(x, P, z, R)`
interface.  The training loop and losses never branch on filter type — swap KF ↔ EKF ↔ UKF
by changing one line.

### Differentiability — non-negotiable

Every operation between `(L_Q, L_R)` and the scalar loss preserves the autograd graph:

- Kalman gain via `torch.cholesky_solve` — never `torch.inverse`.
- Joseph-form covariance update: `P = (I − KH) P (I − KH)ᵀ + K R Kᵀ`.
- UKF sigma-point generation via `torch.linalg.cholesky(P)`, which is differentiable.
- Covariances symmetrised after every step: `P = 0.5 (P + Pᵀ)`.

### `float64` everywhere

All filter computations run in `float64`.  `float32` introduces NIS/NEES bias that
looks like a tuning problem but is really roundoff.  Mixed-precision inputs are cast at
the training-loop boundary.

---

## Module overview

| Module | Contents |
|--------|----------|
| `automatic_kalman.covariance` | [`CholeskyCovariance`](api/covariance.md) — learnable SPD covariance |
| `automatic_kalman.filters.base` | [`Filter`](api/filter-protocol.md) — shared protocol |
| `automatic_kalman.filters.kf` | [`KalmanFilter`](api/kalman-filter.md) |
| `automatic_kalman.filters.ekf` | [`ExtendedKalmanFilter`](api/extended-kalman-filter.md) |
| `automatic_kalman.filters.ukf` | [`UnscentedKalmanFilter`](api/unscented-kalman-filter.md) |
| `automatic_kalman.losses` | [`nis_consistency`](api/losses.md#nis_consistency), [`nees_consistency`](api/losses.md#nees_consistency) |
| `automatic_kalman.train` | [`train`](api/train.md) — filter-agnostic training loop |
| `automatic_kalman.types` | [`FilterOutputs`](api/types.md) — dataclass passed to losses |
| `automatic_kalman.data.generators` | [`LinearGaussianGenerator`](api/generators.md), [`NonlinearGenerator`](api/generators.md), [`TrajectoryDataset`](api/generators.md) |

---

## Running the notebooks

Two end-to-end notebooks live in `notebooks/`:

```bash
uv run jupyter lab notebooks/01_linear_kf_tracking.ipynb
uv run jupyter lab notebooks/02_nonlinear_ekf_ukf_msd.ipynb
```

See the [Examples](examples/linear-kf-tracking.md) section for a full description of
what each notebook demonstrates.
