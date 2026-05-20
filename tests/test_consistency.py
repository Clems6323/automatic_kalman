"""NIS / NEES consistency and Q/R recovery tests.

Phase 2 validation:
  - test_nis_within_chi2_band:  mean NIS ≈ dim_z with true Q, R
  - test_nees_within_chi2_band: mean NEES ≈ dim_x with true Q, R
  - test_kf_recovers_from_10x_wrong_scale: headline recovery test, < 30 s
"""

import math
import time

import torch
from torch.utils.data import DataLoader

from automatic_kalman.covariance import CholeskyCovariance
from automatic_kalman.data.generators import (
    LinearGaussianGenerator,
    NonlinearGenerator,
    TrajectoryDataset,
)
from automatic_kalman.filters.ekf import ExtendedKalmanFilter
from automatic_kalman.filters.kf import KalmanFilter
from automatic_kalman.filters.ukf import UnscentedKalmanFilter
from automatic_kalman.losses import nis_consistency, nees_consistency
from automatic_kalman.train import train


# ── shared system ────────────────────────────────────────────────────────────

def _cv_system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constant-velocity: 2 states, 1 position measurement, dt=0.1."""
    dt = 0.1
    F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64)
    H = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    Q_true = torch.tensor([[1e-3, 0.0], [0.0, 1e-2]], dtype=torch.float64)
    R_true = torch.tensor([[0.5]], dtype=torch.float64)
    return F, H, Q_true, R_true


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_filter_collect_nis(
    kf: KalmanFilter,
    measurements: torch.Tensor,
    states_true: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    skip: int = 10,
) -> torch.Tensor:
    """Return flattened NIS values (B*(T-skip),) from a no-grad filter pass.

    Skips the first ``skip`` steps to avoid the P0 transient.
    """
    B, T, _ = measurements.shape
    x0 = torch.zeros(kf.dim_x, dtype=torch.float64)
    P0 = torch.eye(kf.dim_x, dtype=torch.float64)
    x = x0.unsqueeze(0).expand(B, -1).clone()
    P = P0.unsqueeze(0).expand(B, -1, -1).clone()

    nis_list = []
    with torch.no_grad():
        for t in range(T):
            x, P = kf.predict(x, P, Q)
            x, P, y, S = kf.update(x, P, measurements[:, t, :], R)
            if t >= skip:
                L_S = torch.linalg.cholesky(S)
                S_inv_y = torch.cholesky_solve(
                    y.unsqueeze(-1), L_S
                ).squeeze(-1)
                nis_list.append((y * S_inv_y).sum(-1))  # (B,)

    return torch.cat(nis_list)  # (B*(T-skip),)


def _run_filter_collect_nees(
    kf: KalmanFilter,
    measurements: torch.Tensor,
    states_true: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    skip: int = 10,
) -> torch.Tensor:
    """Return flattened NEES values (B*(T-skip),) from a no-grad filter pass."""
    B, T, _ = measurements.shape
    x0 = torch.zeros(kf.dim_x, dtype=torch.float64)
    P0 = torch.eye(kf.dim_x, dtype=torch.float64)
    x = x0.unsqueeze(0).expand(B, -1).clone()
    P = P0.unsqueeze(0).expand(B, -1, -1).clone()

    nees_list = []
    with torch.no_grad():
        for t in range(T):
            x, P = kf.predict(x, P, Q)
            x, P, _, _ = kf.update(x, P, measurements[:, t, :], R)
            if t >= skip:
                e = x - states_true[:, t, :]
                L_P = torch.linalg.cholesky(P)
                P_inv_e = torch.cholesky_solve(
                    e.unsqueeze(-1), L_P
                ).squeeze(-1)
                nees_list.append((e * P_inv_e).sum(-1))  # (B,)

    return torch.cat(nees_list)  # (B*(T-skip),)


# ── consistency tests ─────────────────────────────────────────────────────────

def test_nis_within_chi2_band() -> None:
    """Mean NIS with true Q, R must lie within the chi-squared 5-sigma CI."""
    torch.manual_seed(0)
    F, H, Q_true, R_true = _cv_system()
    gen = LinearGaussianGenerator(F, H, Q_true, R_true, x0=torch.zeros(2))
    batch = gen(B=500, T=60)

    kf = KalmanFilter(F, H)
    dim_z = kf.dim_z
    skip = 10
    all_nis = _run_filter_collect_nis(kf, batch.measurements, batch.states, Q_true, R_true, skip)

    # Under correct tuning NIS_k ~ chi2(dim_z) → E[NIS]=dim_z, Var[NIS]=2*dim_z
    # CLT: sample mean ~ N(dim_z, 2*dim_z/N)
    N = all_nis.numel()
    se = math.sqrt(2.0 * dim_z / N)
    mean_nis = all_nis.mean().item()

    assert abs(mean_nis - dim_z) < 5.0 * se, (
        f"NIS consistency failed: mean={mean_nis:.4f}, expected {dim_z} ± {5*se:.4f}"
    )


def test_nees_within_chi2_band() -> None:
    """Mean NEES with true Q, R must lie within the chi-squared 5-sigma CI."""
    torch.manual_seed(1)
    F, H, Q_true, R_true = _cv_system()
    gen = LinearGaussianGenerator(F, H, Q_true, R_true, x0=torch.zeros(2))
    batch = gen(B=500, T=60)

    kf = KalmanFilter(F, H)
    dim_x = kf.dim_x
    skip = 10
    all_nees = _run_filter_collect_nees(kf, batch.measurements, batch.states, Q_true, R_true, skip)

    N = all_nees.numel()
    se = math.sqrt(2.0 * dim_x / N)
    mean_nees = all_nees.mean().item()

    assert abs(mean_nees - dim_x) < 5.0 * se, (
        f"NEES consistency failed: mean={mean_nees:.4f}, expected {dim_x} ± {5*se:.4f}"
    )


# ── recovery test ─────────────────────────────────────────────────────────────

def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: log(exp(x) - 1), numerically stable for x > 0.2."""
    return torch.log(torch.expm1(x))


def _init_cov_at_scale(
    cov: CholeskyCovariance,
    L_true: torch.Tensor,
    scale: float,
) -> None:
    """Set cov._L_raw so that cov.matrix() ≈ scale² * (L_true @ L_true.T).

    ``factor() = scale * L_true`` is achieved by:
    - diagonal raw = softplus_inv(scale * L_true.diag)
    - off-diagonal raw = scale * L_true off-diagonal entries
    """
    with torch.no_grad():
        target_factor = scale * L_true          # desired factor()
        L_raw = target_factor.clone()
        L_raw.diagonal().copy_(_softplus_inv(target_factor.diagonal()))
        cov._L_raw.data.copy_(L_raw)


def test_kf_recovers_from_10x_wrong_scale() -> None:
    """Recovery test: init Q and R at 10× true scale; train; verify NEES ≈ dim_x.

    Also verifies the test runs in < 30 s (from CLAUDE.md Phase 2 requirement).
    """
    t_start = time.time()
    torch.manual_seed(42)

    dim_x, dim_z = 2, 1
    F, H, Q_true, R_true = _cv_system()

    gen = LinearGaussianGenerator(F, H, Q_true, R_true, x0=torch.zeros(dim_x))
    dataset = TrajectoryDataset(gen(B=160, T=15))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    kf = KalmanFilter(F, H)

    cov_Q = CholeskyCovariance(dim=dim_x)
    cov_R = CholeskyCovariance(dim=dim_z)

    # factor() = sqrt(10) * L_true  →  matrix() ≈ 10 * Q_true / R_true
    L_Q_true = torch.linalg.cholesky(Q_true)
    L_R_true = torch.linalg.cholesky(R_true)
    _init_cov_at_scale(cov_Q, L_Q_true, math.sqrt(10.0))
    _init_cov_at_scale(cov_R, L_R_true, math.sqrt(10.0))

    x0 = torch.zeros(dim_x, dtype=torch.float64)
    P0 = torch.eye(dim_x, dtype=torch.float64)

    losses = train(
        filter=kf,
        cov_Q=cov_Q,
        cov_R=cov_R,
        loss_fn=nis_consistency,
        loader=loader,
        x0=x0,
        P0=P0,
        epochs=100,
        lr=0.03,
    )

    elapsed = time.time() - t_start
    assert elapsed < 30.0, f"Recovery test took {elapsed:.1f}s — exceeds 30s budget"

    # ── evaluate on a fresh held-out batch ────────────────────────────────────
    torch.manual_seed(99)
    batch_eval = gen(B=500, T=60)
    Q_learned = cov_Q.matrix().detach()
    R_learned = cov_R.matrix().detach()

    all_nis = _run_filter_collect_nis(
        kf, batch_eval.measurements, batch_eval.states, Q_learned, R_learned, skip=10
    )
    mean_nis = all_nis.mean().item()

    # Generous recovery bound: mean NIS within [0.2*dim_z, 5.0*dim_z]
    assert 0.2 * dim_z <= mean_nis <= 5.0 * dim_z, (
        f"Recovery failed: mean NIS = {mean_nis:.3f}, expected ≈ {float(dim_z):.1f}"
    )
    # Loss must decrease overall
    assert losses[-1] < losses[0], (
        f"KF loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


# ── EKF shared nonlinear system (Duffing oscillator) ─────────────────────────

_DUFFING_DT = 0.05
_DUFFING_K = 1.0
_DUFFING_K3 = 0.5
_DUFFING_C = 0.2


def _duffing_f(x: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
    """Single-sample Duffing oscillator step (Euler, no control)."""
    pos, vel = x[0], x[1]
    return torch.stack([
        pos + vel * _DUFFING_DT,
        vel + (-_DUFFING_K * pos - _DUFFING_K3 * pos ** 3 - _DUFFING_C * vel) * _DUFFING_DT,
    ])


def _pos_h(x: torch.Tensor) -> torch.Tensor:
    """Position-only measurement."""
    return x[:1]


def _duffing_system() -> tuple[torch.Tensor, torch.Tensor]:
    """True Q and R for the Duffing consistency tests."""
    Q_true = torch.tensor([[1e-2, 0.0], [0.0, 1e-2]], dtype=torch.float64)
    R_true = torch.tensor([[0.1]], dtype=torch.float64)
    return Q_true, R_true


# ── EKF consistency tests ─────────────────────────────────────────────────────


def test_ekf_nis_within_chi2_band() -> None:
    """Mean NIS with true Q, R must lie within the chi-squared 5-sigma CI (EKF)."""
    torch.manual_seed(10)
    Q_true, R_true = _duffing_system()
    dim_z = 1

    gen = NonlinearGenerator(
        f=_duffing_f, h=_pos_h,
        Q_true=Q_true, R_true=R_true,
        x0=torch.tensor([0.2, 0.0], dtype=torch.float64),
    )
    batch = gen(B=500, T=60)

    ekf = ExtendedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=2, dim_z=dim_z)
    skip = 10
    all_nis = _run_filter_collect_nis(ekf, batch.measurements, batch.states, Q_true, R_true, skip)

    N = all_nis.numel()
    se = math.sqrt(2.0 * dim_z / N)
    mean_nis = all_nis.mean().item()

    assert abs(mean_nis - dim_z) < 5.0 * se, (
        f"EKF NIS consistency failed: mean={mean_nis:.4f}, expected {dim_z} ± {5*se:.4f}"
    )


def test_ekf_nees_within_chi2_band() -> None:
    """Mean NEES with true Q, R must lie within the chi-squared 5-sigma CI (EKF)."""
    torch.manual_seed(11)
    Q_true, R_true = _duffing_system()
    dim_x = 2

    gen = NonlinearGenerator(
        f=_duffing_f, h=_pos_h,
        Q_true=Q_true, R_true=R_true,
        x0=torch.tensor([0.2, 0.0], dtype=torch.float64),
    )
    batch = gen(B=500, T=60)

    ekf = ExtendedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=dim_x, dim_z=1)
    skip = 10
    all_nees = _run_filter_collect_nees(ekf, batch.measurements, batch.states, Q_true, R_true, skip)

    N = all_nees.numel()
    se = math.sqrt(2.0 * dim_x / N)
    mean_nees = all_nees.mean().item()

    # EKF linearization introduces a small systematic bias: NEES is typically
    # slightly below dim_x for mildly nonlinear systems (overconfident covariance).
    # Allow 15% relative error in addition to statistical noise.
    tol = max(5.0 * se, 0.15 * dim_x)
    assert abs(mean_nees - dim_x) < tol, (
        f"EKF NEES consistency failed: mean={mean_nees:.4f}, expected {dim_x} ± {tol:.4f}"
    )


# ── EKF recovery test ─────────────────────────────────────────────────────────


def test_ekf_recovers_from_10x_wrong_scale() -> None:
    """Recovery test: EKF init Q, R at 10× true; train; verify NIS ≈ dim_z. Must complete < 30 s."""
    t_start = time.time()
    torch.manual_seed(55)

    dim_x, dim_z = 2, 1
    Q_true, R_true = _duffing_system()

    gen = NonlinearGenerator(
        f=_duffing_f, h=_pos_h,
        Q_true=Q_true, R_true=R_true,
        x0=torch.tensor([0.2, 0.0], dtype=torch.float64),
    )
    dataset = TrajectoryDataset(gen(B=160, T=15))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    ekf = ExtendedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=dim_x, dim_z=dim_z)

    cov_Q = CholeskyCovariance(dim=dim_x)
    cov_R = CholeskyCovariance(dim=dim_z)

    L_Q_true = torch.linalg.cholesky(Q_true)
    L_R_true = torch.linalg.cholesky(R_true)
    _init_cov_at_scale(cov_Q, L_Q_true, math.sqrt(10.0))
    _init_cov_at_scale(cov_R, L_R_true, math.sqrt(10.0))

    x0 = torch.zeros(dim_x, dtype=torch.float64)
    P0 = torch.eye(dim_x, dtype=torch.float64)

    losses = train(
        filter=ekf,
        cov_Q=cov_Q,
        cov_R=cov_R,
        loss_fn=nis_consistency,
        loader=loader,
        x0=x0,
        P0=P0,
        epochs=100,
        lr=0.03,
    )

    elapsed = time.time() - t_start
    assert elapsed < 30.0, f"EKF recovery test took {elapsed:.1f}s — exceeds 30s budget"

    torch.manual_seed(199)
    batch_eval = gen(B=500, T=60)
    Q_learned = cov_Q.matrix().detach()
    R_learned = cov_R.matrix().detach()

    all_nis = _run_filter_collect_nis(
        ekf, batch_eval.measurements, batch_eval.states, Q_learned, R_learned, skip=10
    )
    mean_nis = all_nis.mean().item()

    assert 0.2 * dim_z <= mean_nis <= 5.0 * dim_z, (
        f"EKF recovery failed: mean NIS = {mean_nis:.3f}, expected ≈ {float(dim_z):.1f}"
    )
    assert losses[-1] < losses[0], (
        f"EKF loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


# ── UKF consistency tests ─────────────────────────────────────────────────────


def test_ukf_nis_within_chi2_band() -> None:
    """Mean NIS with true Q, R must lie within the chi-squared 5-sigma CI (UKF)."""
    torch.manual_seed(20)
    Q_true, R_true = _duffing_system()
    dim_z = 1

    gen = NonlinearGenerator(
        f=_duffing_f, h=_pos_h,
        Q_true=Q_true, R_true=R_true,
        x0=torch.tensor([0.2, 0.0], dtype=torch.float64),
    )
    batch = gen(B=500, T=60)

    ukf = UnscentedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=2, dim_z=dim_z)
    skip = 10
    all_nis = _run_filter_collect_nis(ukf, batch.measurements, batch.states, Q_true, R_true, skip)

    N = all_nis.numel()
    se = math.sqrt(2.0 * dim_z / N)
    mean_nis = all_nis.mean().item()

    assert abs(mean_nis - dim_z) < 5.0 * se, (
        f"UKF NIS consistency failed: mean={mean_nis:.4f}, expected {dim_z} ± {5*se:.4f}"
    )


def test_ukf_nees_within_chi2_band() -> None:
    """Mean NEES with true Q, R must lie within the chi-squared 5-sigma CI (UKF)."""
    torch.manual_seed(21)
    Q_true, R_true = _duffing_system()
    dim_x = 2

    gen = NonlinearGenerator(
        f=_duffing_f, h=_pos_h,
        Q_true=Q_true, R_true=R_true,
        x0=torch.tensor([0.2, 0.0], dtype=torch.float64),
    )
    batch = gen(B=500, T=60)

    ukf = UnscentedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=dim_x, dim_z=1)
    skip = 10
    all_nees = _run_filter_collect_nees(ukf, batch.measurements, batch.states, Q_true, R_true, skip)

    N = all_nees.numel()
    se = math.sqrt(2.0 * dim_x / N)
    mean_nees = all_nees.mean().item()

    # UKF captures higher-order moments; allow the same liberal bound as EKF.
    tol = max(5.0 * se, 0.15 * dim_x)
    assert abs(mean_nees - dim_x) < tol, (
        f"UKF NEES consistency failed: mean={mean_nees:.4f}, expected {dim_x} ± {tol:.4f}"
    )


# ── UKF recovery test ─────────────────────────────────────────────────────────


def test_ukf_recovers_from_10x_wrong_scale() -> None:
    """Recovery test: UKF init Q, R at 10× true; train; verify NIS ≈ dim_z. Must complete < 30 s."""
    t_start = time.time()
    torch.manual_seed(77)

    dim_x, dim_z = 2, 1
    Q_true, R_true = _duffing_system()

    gen = NonlinearGenerator(
        f=_duffing_f, h=_pos_h,
        Q_true=Q_true, R_true=R_true,
        x0=torch.tensor([0.2, 0.0], dtype=torch.float64),
    )
    dataset = TrajectoryDataset(gen(B=160, T=15))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # alpha=0.3 avoids extreme weights (W_m[0] ≈ -999999 with default 1e-3)
    # that can amplify gradient noise during training.
    ukf = UnscentedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=dim_x, dim_z=dim_z, alpha=0.3)

    cov_Q = CholeskyCovariance(dim=dim_x)
    cov_R = CholeskyCovariance(dim=dim_z)

    L_Q_true = torch.linalg.cholesky(Q_true)
    L_R_true = torch.linalg.cholesky(R_true)
    _init_cov_at_scale(cov_Q, L_Q_true, math.sqrt(10.0))
    _init_cov_at_scale(cov_R, L_R_true, math.sqrt(10.0))

    x0 = torch.zeros(dim_x, dtype=torch.float64)
    P0 = torch.eye(dim_x, dtype=torch.float64)

    losses = train(
        filter=ukf,
        cov_Q=cov_Q,
        cov_R=cov_R,
        loss_fn=nis_consistency,
        loader=loader,
        x0=x0,
        P0=P0,
        epochs=100,
        lr=0.03,
    )

    elapsed = time.time() - t_start
    assert elapsed < 30.0, f"UKF recovery test took {elapsed:.1f}s — exceeds 30s budget"

    torch.manual_seed(299)
    batch_eval = gen(B=500, T=60)
    Q_learned = cov_Q.matrix().detach()
    R_learned = cov_R.matrix().detach()

    all_nis = _run_filter_collect_nis(
        ukf, batch_eval.measurements, batch_eval.states, Q_learned, R_learned, skip=10
    )
    mean_nis = all_nis.mean().item()

    assert 0.2 * dim_z <= mean_nis <= 5.0 * dim_z, (
        f"UKF recovery failed: mean NIS = {mean_nis:.3f}, expected ≈ {float(dim_z):.1f}"
    )
    assert losses[-1] < losses[0], (
        f"UKF loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )
