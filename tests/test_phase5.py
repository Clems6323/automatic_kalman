"""Phase 5 integration and hardening tests.

Covers:
  - float64 audit: all training-graph tensors must be float64
  - Condition-number monitor: warning fires above threshold, silent otherwise
  - Multi-filter comparison: KF / EKF / UKF agree within 1e-6 on linear-Gaussian data
  - Performance gate: KF + EKF + UKF recovery runs combined < 30 s on CPU
  - GPU smoke test: KF training on CUDA matches CPU within tolerance
"""

import logging
import math
import time

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from automatic_kalman.covariance import CholeskyCovariance
from automatic_kalman.data.generators import LinearGaussianGenerator, NonlinearGenerator, TrajectoryDataset
from automatic_kalman.filters.ekf import ExtendedKalmanFilter
from automatic_kalman.filters.kf import KalmanFilter
from automatic_kalman.filters.ukf import UnscentedKalmanFilter
from automatic_kalman.losses import nis_consistency
from automatic_kalman.train import train


# ── shared fixtures ───────────────────────────────────────────────────────────

_DT = 0.1
_F = torch.tensor([[1.0, _DT], [0.0, 1.0]], dtype=torch.float64)
_H = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
_Q_TRUE = torch.tensor([[1e-3, 0.0], [0.0, 1e-2]], dtype=torch.float64)
_R_TRUE = torch.tensor([[0.5]], dtype=torch.float64)

_DUFFING_DT = 0.05
_DUFFING_K = 1.0
_DUFFING_K3 = 0.5
_DUFFING_C = 0.2


def _duffing_f(x: Tensor, u: Tensor | None) -> Tensor:
    pos, vel = x[0], x[1]
    return torch.stack([
        pos + vel * _DUFFING_DT,
        vel + (-_DUFFING_K * pos - _DUFFING_K3 * pos ** 3 - _DUFFING_C * vel) * _DUFFING_DT,
    ])


def _pos_h(x: Tensor) -> Tensor:
    return x[:1]


def _softplus_inv(x: Tensor) -> Tensor:
    return torch.log(torch.expm1(x))


def _init_cov_at_scale(cov: CholeskyCovariance, L_true: Tensor, scale: float) -> None:
    with torch.no_grad():
        target_factor = scale * L_true
        L_raw = target_factor.clone()
        L_raw.diagonal().copy_(_softplus_inv(target_factor.diagonal()))
        cov._L_raw.data.copy_(L_raw)


# ── float64 audit ─────────────────────────────────────────────────────────────


def test_cholesky_covariance_is_float64() -> None:
    """CholeskyCovariance.factor() and .matrix() must always be float64."""
    for dim in (1, 2, 5):
        cov = CholeskyCovariance(dim=dim)
        assert cov.factor().dtype == torch.float64, f"factor() dtype wrong for dim={dim}"
        assert cov.matrix().dtype == torch.float64, f"matrix() dtype wrong for dim={dim}"


def test_training_graph_tensors_are_float64() -> None:
    """All filter state, covariance, innovation, and innovation-cov tensors must be float64."""
    torch.manual_seed(0)
    gen = LinearGaussianGenerator(_F, _H, _Q_TRUE, _R_TRUE, x0=torch.zeros(2))
    dataset = TrajectoryDataset(gen(B=4, T=5))
    loader = DataLoader(dataset, batch_size=4)

    kf = KalmanFilter(_F, _H)
    cov_Q = CholeskyCovariance(dim=2)
    cov_R = CholeskyCovariance(dim=1)

    Q = cov_Q.matrix()
    R = cov_R.matrix()
    assert Q.dtype == torch.float64, f"Q dtype: {Q.dtype}"
    assert R.dtype == torch.float64, f"R dtype: {R.dtype}"

    x0 = torch.zeros(2, dtype=torch.float64)
    P0 = torch.eye(2, dtype=torch.float64)

    for z_batch, _ in loader:
        z_batch = z_batch.double()
        B, T, _ = z_batch.shape
        x = x0.unsqueeze(0).expand(B, -1).clone()
        P = P0.unsqueeze(0).expand(B, -1, -1).clone()
        for t in range(T):
            x, P = kf.predict(x, P, Q)
            x, P, y, S = kf.update(x, P, z_batch[:, t, :], R)
        break

    for name, tensor in [("x", x), ("P", P), ("y", y), ("S", S)]:
        assert tensor.dtype == torch.float64, f"{name} dtype: {tensor.dtype}"


# ── condition-number monitor ───────────────────────────────────────────────────


def test_cond_monitor_no_spurious_warning(caplog: pytest.LogCaptureFixture) -> None:
    """No WARNING about condition numbers during well-conditioned training."""
    torch.manual_seed(0)
    gen = LinearGaussianGenerator(_F, _H, _Q_TRUE, _R_TRUE, x0=torch.zeros(2))
    dataset = TrajectoryDataset(gen(B=8, T=10))
    loader = DataLoader(dataset, batch_size=8)

    kf = KalmanFilter(_F, _H)
    cov_Q = CholeskyCovariance(dim=2)
    cov_R = CholeskyCovariance(dim=1)
    x0 = torch.zeros(2, dtype=torch.float64)
    P0 = torch.eye(2, dtype=torch.float64)

    with caplog.at_level(logging.WARNING, logger="automatic_kalman.train"):
        train(
            filter=kf,
            cov_Q=cov_Q,
            cov_R=cov_R,
            loss_fn=nis_consistency,
            loader=loader,
            x0=x0,
            P0=P0,
            epochs=5,
            lr=0.01,
        )

    cond_warnings = [r for r in caplog.records if "condition number" in r.message.lower()]
    assert len(cond_warnings) == 0, (
        f"Unexpected condition-number warnings: {[r.message for r in cond_warnings]}"
    )


def test_cond_monitor_threshold_configurable(caplog: pytest.LogCaptureFixture) -> None:
    """Setting cond_warn_threshold=0 must trigger a WARNING on every epoch."""
    torch.manual_seed(0)
    gen = LinearGaussianGenerator(_F, _H, _Q_TRUE, _R_TRUE, x0=torch.zeros(2))
    dataset = TrajectoryDataset(gen(B=4, T=5))
    loader = DataLoader(dataset, batch_size=4)

    kf = KalmanFilter(_F, _H)
    cov_Q = CholeskyCovariance(dim=2)
    cov_R = CholeskyCovariance(dim=1)
    x0 = torch.zeros(2, dtype=torch.float64)
    P0 = torch.eye(2, dtype=torch.float64)

    with caplog.at_level(logging.WARNING, logger="automatic_kalman.train"):
        train(
            filter=kf,
            cov_Q=cov_Q,
            cov_R=cov_R,
            loss_fn=nis_consistency,
            loader=loader,
            x0=x0,
            P0=P0,
            epochs=3,
            lr=0.0,
            cond_warn_threshold=0.0,  # always fires
        )

    cond_warnings = [r for r in caplog.records if "condition number" in r.message.lower()]
    assert len(cond_warnings) == 3, (
        f"Expected 3 warnings (one per epoch), got {len(cond_warnings)}"
    )


# ── multi-filter comparison ───────────────────────────────────────────────────


def test_kf_ekf_ukf_agree_on_linear_gaussian() -> None:
    """KF, EKF, and UKF must agree within 1e-6 on linear-Gaussian trajectories.

    Uses linear f and h so all three filters are algebraically equivalent.
    Validates that train.py is truly filter-agnostic and that all three
    implementations share a consistent numerical baseline.
    """
    Q = torch.tensor([[1e-3, 0.0], [0.0, 1e-2]], dtype=torch.float64)
    R = torch.tensor([[0.5]], dtype=torch.float64)

    def f_lin(x: Tensor, u: Tensor | None) -> Tensor:
        return _F @ x

    def h_lin(x: Tensor) -> Tensor:
        return _H @ x

    kf = KalmanFilter(_F, _H)
    ekf = ExtendedKalmanFilter(f=f_lin, h=h_lin, dim_x=2, dim_z=1)
    # alpha=0.3 avoids the catastrophic cancellation of the default alpha=1e-3
    # weights while remaining algebraically equivalent to KF for linear f/h.
    ukf = UnscentedKalmanFilter(f=f_lin, h=h_lin, dim_x=2, dim_z=1, alpha=0.3)

    B, T = 4, 30
    torch.manual_seed(7)
    z_seq = torch.randn(B, T, 1, dtype=torch.float64)

    x0 = torch.zeros(B, 2, dtype=torch.float64)
    P0 = torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).clone()
    x_k, P_k = x0.clone(), P0.clone()
    x_e, P_e = x0.clone(), P0.clone()
    x_u, P_u = x0.clone(), P0.clone()

    with torch.no_grad():
        for t in range(T):
            z_t = z_seq[:, t, :]
            x_k, P_k = kf.predict(x_k, P_k, Q)
            x_k, P_k, y_k, S_k = kf.update(x_k, P_k, z_t, R)
            x_e, P_e = ekf.predict(x_e, P_e, Q)
            x_e, P_e, y_e, S_e = ekf.update(x_e, P_e, z_t, R)
            x_u, P_u = ukf.predict(x_u, P_u, Q)
            x_u, P_u, y_u, S_u = ukf.update(x_u, P_u, z_t, R)

            assert torch.allclose(x_e, x_k, atol=1e-6), f"EKF/KF x mismatch at step {t}"
            assert torch.allclose(x_u, x_k, atol=1e-6), f"UKF/KF x mismatch at step {t}"
            assert torch.allclose(P_e, P_k, atol=1e-6), f"EKF/KF P mismatch at step {t}"
            assert torch.allclose(P_u, P_k, atol=1e-6), f"UKF/KF P mismatch at step {t}"
            assert torch.allclose(y_e, y_k, atol=1e-6), f"EKF/KF y mismatch at step {t}"
            assert torch.allclose(y_u, y_k, atol=1e-6), f"UKF/KF y mismatch at step {t}"
            assert torch.allclose(S_e, S_k, atol=1e-6), f"EKF/KF S mismatch at step {t}"
            assert torch.allclose(S_u, S_k, atol=1e-6), f"UKF/KF S mismatch at step {t}"


# ── performance gate ──────────────────────────────────────────────────────────


def test_combined_recovery_under_30s() -> None:
    """KF + EKF + UKF covariance recovery (B=80, T=10, 60 epochs each) must complete in < 30 s.

    Uses reduced parameters compared to the full recovery tests so the combined
    run is well within budget.  The three individual tests in test_consistency.py
    verify correctness with larger datasets; this test guards against throughput
    regressions that would make the framework impractical.
    """
    Q_true_duf = torch.tensor([[1e-2, 0.0], [0.0, 1e-2]], dtype=torch.float64)
    R_true_duf = torch.tensor([[0.1]], dtype=torch.float64)

    t_start = time.time()

    # ── KF ────────────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    gen_kf = LinearGaussianGenerator(_F, _H, _Q_TRUE, _R_TRUE, x0=torch.zeros(2))
    loader_kf = DataLoader(TrajectoryDataset(gen_kf(B=80, T=10)), batch_size=32, shuffle=True)
    kf = KalmanFilter(_F, _H)
    cov_Q_kf = CholeskyCovariance(dim=2)
    cov_R_kf = CholeskyCovariance(dim=1)
    _init_cov_at_scale(cov_Q_kf, torch.linalg.cholesky(_Q_TRUE), math.sqrt(10.0))
    _init_cov_at_scale(cov_R_kf, torch.linalg.cholesky(_R_TRUE), math.sqrt(10.0))
    train(
        filter=kf, cov_Q=cov_Q_kf, cov_R=cov_R_kf,
        loss_fn=nis_consistency, loader=loader_kf,
        x0=torch.zeros(2, dtype=torch.float64),
        P0=torch.eye(2, dtype=torch.float64),
        epochs=60, lr=0.03,
    )

    # ── EKF ───────────────────────────────────────────────────────────────────
    torch.manual_seed(43)
    gen_ekf = NonlinearGenerator(
        f=_duffing_f, h=_pos_h, Q_true=Q_true_duf, R_true=R_true_duf,
        x0=torch.tensor([0.2, 0.0], dtype=torch.float64),
    )
    loader_ekf = DataLoader(TrajectoryDataset(gen_ekf(B=80, T=10)), batch_size=32, shuffle=True)
    ekf = ExtendedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=2, dim_z=1)
    cov_Q_ekf = CholeskyCovariance(dim=2)
    cov_R_ekf = CholeskyCovariance(dim=1)
    _init_cov_at_scale(cov_Q_ekf, torch.linalg.cholesky(Q_true_duf), math.sqrt(10.0))
    _init_cov_at_scale(cov_R_ekf, torch.linalg.cholesky(R_true_duf), math.sqrt(10.0))
    train(
        filter=ekf, cov_Q=cov_Q_ekf, cov_R=cov_R_ekf,
        loss_fn=nis_consistency, loader=loader_ekf,
        x0=torch.zeros(2, dtype=torch.float64),
        P0=torch.eye(2, dtype=torch.float64),
        epochs=60, lr=0.03,
    )

    # ── UKF ───────────────────────────────────────────────────────────────────
    torch.manual_seed(44)
    gen_ukf = NonlinearGenerator(
        f=_duffing_f, h=_pos_h, Q_true=Q_true_duf, R_true=R_true_duf,
        x0=torch.tensor([0.2, 0.0], dtype=torch.float64),
    )
    loader_ukf = DataLoader(TrajectoryDataset(gen_ukf(B=80, T=10)), batch_size=32, shuffle=True)
    ukf = UnscentedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=2, dim_z=1, alpha=0.3)
    cov_Q_ukf = CholeskyCovariance(dim=2)
    cov_R_ukf = CholeskyCovariance(dim=1)
    _init_cov_at_scale(cov_Q_ukf, torch.linalg.cholesky(Q_true_duf), math.sqrt(10.0))
    _init_cov_at_scale(cov_R_ukf, torch.linalg.cholesky(R_true_duf), math.sqrt(10.0))
    train(
        filter=ukf, cov_Q=cov_Q_ukf, cov_R=cov_R_ukf,
        loss_fn=nis_consistency, loader=loader_ukf,
        x0=torch.zeros(2, dtype=torch.float64),
        P0=torch.eye(2, dtype=torch.float64),
        epochs=60, lr=0.03,
    )

    elapsed = time.time() - t_start
    assert elapsed < 30.0, (
        f"Combined KF+EKF+UKF recovery took {elapsed:.1f}s — exceeds 30s performance gate"
    )


# ── GPU smoke test ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kf_gpu_matches_cpu() -> None:
    """KF training on GPU must produce losses that agree with CPU within 1e-4.

    Both runs use the same seed and identical data; any discrepancy reflects
    floating-point reordering between CPU and CUDA, not a correctness bug.
    """
    torch.manual_seed(0)
    gen = LinearGaussianGenerator(_F, _H, _Q_TRUE, _R_TRUE, x0=torch.zeros(2))
    dataset = TrajectoryDataset(gen(B=16, T=10))
    # Use a fixed loader (no shuffle) so CPU and GPU see identical batches.
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    x0 = torch.zeros(2, dtype=torch.float64)
    P0 = torch.eye(2, dtype=torch.float64)

    def _run(device: torch.device) -> list[float]:
        torch.manual_seed(1)
        cov_Q = CholeskyCovariance(dim=2).to(device)
        cov_R = CholeskyCovariance(dim=1).to(device)
        kf = KalmanFilter(_F.to(device), _H.to(device))
        return train(
            filter=kf, cov_Q=cov_Q, cov_R=cov_R,
            loss_fn=nis_consistency, loader=loader,
            x0=x0, P0=P0,
            epochs=10, lr=0.01,
        )

    cpu_losses = _run(torch.device("cpu"))
    gpu_losses = _run(torch.device("cuda"))

    for t, (lc, lg) in enumerate(zip(cpu_losses, gpu_losses)):
        assert abs(lc - lg) < 1e-4, (
            f"Epoch {t}: CPU loss={lc:.6f} vs GPU loss={lg:.6f} differ by {abs(lc-lg):.2e}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ekf_gpu_forward_pass() -> None:
    """EKF forward pass (no training) on GPU must not raise and produce float64 outputs."""
    Q = _Q_TRUE.cuda()
    R = _R_TRUE.cuda()

    ekf = ExtendedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=2, dim_z=1)
    B, T = 4, 5
    torch.manual_seed(0)
    z_seq = torch.randn(B, T, 1, dtype=torch.float64, device="cuda")
    x = torch.zeros(B, 2, dtype=torch.float64, device="cuda")
    P = torch.eye(2, dtype=torch.float64, device="cuda").unsqueeze(0).expand(B, -1, -1).clone()

    with torch.no_grad():
        for t in range(T):
            x, P = ekf.predict(x, P, Q)
            x, P, y, S = ekf.update(x, P, z_seq[:, t, :], R)

    assert x.device.type == "cuda"
    assert x.dtype == torch.float64
    assert P.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ukf_gpu_forward_pass() -> None:
    """UKF forward pass on GPU must not raise and produce float64 outputs."""
    Q = _Q_TRUE.cuda()
    R = _R_TRUE.cuda()

    ukf = UnscentedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=2, dim_z=1, alpha=0.3)
    B, T = 4, 5
    torch.manual_seed(0)
    z_seq = torch.randn(B, T, 1, dtype=torch.float64, device="cuda")
    x = torch.zeros(B, 2, dtype=torch.float64, device="cuda")
    P = torch.eye(2, dtype=torch.float64, device="cuda").unsqueeze(0).expand(B, -1, -1).clone()

    with torch.no_grad():
        for t in range(T):
            x, P = ukf.predict(x, P, Q)
            x, P, y, S = ukf.update(x, P, z_seq[:, t, :], R)

    assert x.device.type == "cuda"
    assert x.dtype == torch.float64
    assert P.dtype == torch.float64
