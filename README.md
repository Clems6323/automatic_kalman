# Automatic Kalman

Learn the process-noise covariance **Q** and measurement-noise covariance **R** of a
Bayesian filter by gradient descent.

Supports three filter families sharing a common differentiable interface:

| Filter | Dynamics | Jacobians |
|--------|----------|-----------|
| **KF** | Linear | n/a |
| **EKF** | Nonlinear | `torch.func.jacrev` (or analytic opt-in) |
| **UKF** | Nonlinear | None — Merwe scaled sigma points |

`Q` and `R` are Cholesky-parameterised (`L @ L.T + ε·I`) so they stay
symmetric positive-definite throughout training.  Training minimises an NIS or
NEES consistency loss with a log-det regulariser.

## Quickstart

```bash
uv sync
uv run pytest
```

## Lint / type-check

```bash
uv run ruff check src/
uv run mypy src/
```

## Project layout

```
src/automatic_kalman/
  filters/          KF, EKF, UKF — all implement the Filter protocol
  covariance.py     CholeskyCovariance module (L_Q, L_R parameters)
  losses.py         nis_consistency, nees_consistency
  train.py          Filter-agnostic training loop
  data/             Synthetic trajectory generators
baselines/          NumPy reference implementations — NOT differentiable
tests/              pytest suite (gradcheck, baseline parity, consistency)
notebooks/          End-to-end use-case demonstrations
my_personnal_experiments/  Original proof-of-concept notebooks
```

## GPU note

Configured for CUDA 12.8 (RTX 3060 Ti).  All training uses `float64` — set
`torch.set_default_dtype(torch.float64)` at the top of your script or notebook.
