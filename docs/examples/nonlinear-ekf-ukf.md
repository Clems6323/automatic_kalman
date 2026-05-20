# Nonlinear EKF vs UKF

**Notebook:** `notebooks/02_nonlinear_ekf_ukf_msd.ipynb`

This notebook demonstrates covariance learning on a nonlinear mechanical
system — the Duffing oscillator (a mass-spring-damper with cubic
stiffness).  It trains both `ExtendedKalmanFilter` and
`UnscentedKalmanFilter` on the same data and compares their learned
covariances and calibration.

---

## System

The **Duffing oscillator** is a nonlinear spring-mass system where the
restoring force has a cubic term:

```
ẍ + C ẋ + K x + K₃ x³ = 0
```

Discrete-time Euler integration with `dt = 0.05`:

```python
def f(x, u):
    pos, vel = x[0], x[1]
    acc = -K * pos - K3 * pos**3 - C * vel
    return torch.stack([pos + dt * vel, vel + dt * acc])

def h(x):
    return x[:1]   # observe position only
```

Parameters: `K=1.0`, `K₃=0.5`, `C=0.2`, initial state `x0 = [0.3, 0.0]`.

True covariances:

```python
Q_true = diag([1e-2, 1e-2])
R_true = [[0.1]]
```

---

## What the notebook covers

### Data generation

200 training trajectories of length 60 and 300 held-out evaluation
trajectories of length 100, generated with `NonlinearGenerator`.

### EKF training

An `ExtendedKalmanFilter` with autodiff Jacobians (no analytic `F_fn`,
`H_fn`) is trained for 150 epochs with the NIS consistency loss,
`lr=0.03`, batch size 20.

### UKF training

An `UnscentedKalmanFilter` (`alpha=0.3`) is trained on the same
training data for 150 epochs with the same hyperparameters.

### Plots

**Loss curves** — EKF and UKF training loss on the same axes, showing
that both converge.  The UKF often converges to a slightly lower final
loss on this nonlinear system because it captures higher-order moments.

**Q/R recovery bar chart** — diagonal entries of learned Q and R for
both filters, compared to the true values.

**NIS side-by-side** — per-step NIS on the held-out evaluation set for
EKF and UKF with 95% chi-squared band.  Both filters land within the
band after training.

**State estimates (2 × 2 grid)** — position and velocity estimates for
both EKF and UKF with 3σ shaded uncertainty bands, overlaid on the
true trajectory.

**Summary table:**

| | Q diag (learned) | R (learned) | Mean NIS | In chi-sq band |
|--|--------|--------|--------|--------|
| EKF | ~`[0.010, 0.010]` | ~`[0.10]` | ≈ 1.0 | Yes |
| UKF | ~`[0.010, 0.010]` | ~`[0.10]` | ≈ 1.0 | Yes |

### Discussion cell

The notebook includes a markdown cell discussing:

- When EKF and UKF agree: in the near-linear regime (small oscillation
  amplitude) their estimates are nearly identical because the cubic term
  contributes little.
- When they diverge: at high energy / large amplitude, the cubic term
  makes `f` significantly nonlinear.  The EKF Jacobian linearises around
  the current estimate and may underestimate `P` in highly curved regions.
  The UKF sigma-point propagation captures the cubic effect and keeps `P`
  better calibrated, visible as a lower NIS variance.
- What the NIS score reveals: a consistently high NIS (> chi-sq upper
  bound) indicates Q/R under-estimated; a consistently low NIS indicates
  over-inflation.

---

## Running the notebook

```bash
uv run jupyter lab notebooks/02_nonlinear_ekf_ukf_msd.ipynb
```

Runs top to bottom without errors.  Training time is approximately
3–5 minutes on CPU for both filters combined.

---

## Key code fragments

### EKF setup

```python
from automatic_kalman.filters.ekf import ExtendedKalmanFilter
from automatic_kalman.covariance import CholeskyCovariance

ekf   = ExtendedKalmanFilter(f=f, h=h, dim_x=2, dim_z=1)
cov_Q = CholeskyCovariance(dim=2)
cov_R = CholeskyCovariance(dim=1)
```

### UKF setup

```python
from automatic_kalman.filters.ukf import UnscentedKalmanFilter

ukf   = UnscentedKalmanFilter(f=f, h=h, dim_x=2, dim_z=1, alpha=0.3)
cov_Q_ukf = CholeskyCovariance(dim=2)
cov_R_ukf = CholeskyCovariance(dim=1)
```

### Shared training call

```python
from automatic_kalman.train import train
from automatic_kalman.losses import nis_consistency

for name, flt, cQ, cR in [
    ("EKF", ekf, cov_Q, cov_R),
    ("UKF", ukf, cov_Q_ukf, cov_R_ukf),
]:
    losses = train(
        filter=flt, cov_Q=cQ, cov_R=cR,
        loss_fn=nis_consistency,
        loader=train_loader,
        x0=x0, P0=P0,
        epochs=150, lr=0.03,
    )
```

Note that `train()` is called identically for both filters — no changes
to the training loop for a different filter type.
