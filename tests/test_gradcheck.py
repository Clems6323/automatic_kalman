"""Autograd correctness tests — Phase 1: CholeskyCovariance; Phase 2+: filter losses."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from automatic_kalman.covariance import CholeskyCovariance
from automatic_kalman.filters.ekf import ExtendedKalmanFilter
from automatic_kalman.filters.kf import KalmanFilter
from automatic_kalman.filters.ukf import UnscentedKalmanFilter
from automatic_kalman.losses import nees_consistency, nis_consistency
from automatic_kalman.types import FilterOutputs


# ---------------------------------------------------------------------------
# CholeskyCovariance unit tests
# ---------------------------------------------------------------------------


def test_cholesky_covariance_spd() -> None:
    """matrix() must be symmetric positive-definite."""
    cov = CholeskyCovariance(dim=4)
    M = cov.matrix()

    assert torch.allclose(M, M.T, atol=1e-12), "matrix() is not symmetric"
    eigvals = torch.linalg.eigvalsh(M)
    assert (eigvals > 0).all(), f"Non-positive eigenvalue found: {eigvals}"


def test_cholesky_covariance_diagonal_positive() -> None:
    """factor() diagonal must stay strictly positive even with negative raw values."""
    cov = CholeskyCovariance(dim=3)

    assert (cov.factor().diagonal() > 0).all(), "Default diagonal not positive"

    # Force raw diagonal to large negative values — softplus must still give >0
    with torch.no_grad():
        cov._L_raw.diagonal().fill_(-50.0)
    assert (cov.factor().diagonal() > 0).all(), "Diagonal not positive after negative init"


def test_cholesky_covariance_identity_init() -> None:
    """Default init should give matrix() = (1 + eps) * I exactly."""
    dim = 3
    cov = CholeskyCovariance(dim=dim)
    M = cov.matrix()
    expected = (1.0 + cov.eps) * torch.eye(dim, dtype=torch.float64)
    assert torch.allclose(M, expected, atol=1e-12), (
        f"Identity init failed.\nExpected:\n{expected}\nGot:\n{M}"
    )


def test_cholesky_covariance_off_diagonal_free() -> None:
    """Off-diagonal entries of factor() should equal _L_raw off-diagonals exactly."""
    dim = 3
    cov = CholeskyCovariance(dim=dim)
    with torch.no_grad():
        cov._L_raw[1, 0] = 2.5
        cov._L_raw[2, 0] = -1.3
    L = cov.factor()
    assert abs(L[1, 0].item() - 2.5) < 1e-12
    assert abs(L[2, 0].item() - (-1.3)) < 1e-12


def test_cholesky_covariance_gradcheck() -> None:
    """torch.autograd.gradcheck on the matrix() transformation."""
    dim = 2
    eps_jitter = 1e-6

    def matrix_from_raw(L_raw: Tensor) -> Tensor:
        L = L_raw.tril()
        off_diag = L.tril(-1)
        diag = torch.diag(F.softplus(L.diagonal()))
        factor = off_diag + diag
        I = torch.eye(dim, dtype=L_raw.dtype, device=L_raw.device)
        return factor @ factor.T + eps_jitter * I

    # Start near identity so the Jacobian is well-conditioned
    init = math.log(math.exp(1.0) - 1.0)
    L_raw = torch.eye(dim, dtype=torch.float64, requires_grad=True) * init

    assert torch.autograd.gradcheck(matrix_from_raw, (L_raw,), eps=1e-6, rtol=1e-3)


# ---------------------------------------------------------------------------
# KF loss gradcheck helpers
# ---------------------------------------------------------------------------

_DIM_X = 2
_DIM_Z = 1
_T = 3
_B = 2
_DT = 0.1

_F = torch.tensor([[1.0, _DT], [0.0, 1.0]], dtype=torch.float64)
_H = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
_SOFTPLUS_INV_1 = math.log(math.exp(1.0) - 1.0)


def _build_spd(L_raw: Tensor, dim: int, eps: float = 1e-6) -> Tensor:
    """Reconstruct an SPD matrix from a raw lower-triangular parameter."""
    L = L_raw.tril()
    factor = L.tril(-1) + torch.diag(F.softplus(L.diagonal()))
    return factor @ factor.T + eps * torch.eye(dim, dtype=L_raw.dtype, device=L_raw.device)


def _run_kf(
    L_Q_raw: Tensor,
    L_R_raw: Tensor,
    z_seq: Tensor,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Run a T-step KF and return (innovations, inn_covs, states, state_covs)."""
    kf = KalmanFilter(_F, _H)
    Q = _build_spd(L_Q_raw, _DIM_X)
    R = _build_spd(L_R_raw, _DIM_Z)

    B = z_seq.shape[0]
    x = torch.zeros(B, _DIM_X, dtype=L_Q_raw.dtype)
    P = torch.eye(_DIM_X, dtype=L_Q_raw.dtype).unsqueeze(0).expand(B, -1, -1).clone()

    innovations, inn_covs, states, state_covs = [], [], [], []
    for t in range(_T):
        x, P = kf.predict(x, P, Q)
        x, P, y, S = kf.update(x, P, z_seq[:, t, :], R)
        innovations.append(y)
        inn_covs.append(S)
        states.append(x)
        state_covs.append(P)

    return innovations, inn_covs, states, state_covs


# Fixed measurements — not part of what we differentiate through.
torch.manual_seed(7)
_Z_SEQ = torch.randn(_B, _T, _DIM_Z, dtype=torch.float64)
_X_TRUE = torch.randn(_B, _T, _DIM_X, dtype=torch.float64)


# ---------------------------------------------------------------------------
# NIS gradcheck
# ---------------------------------------------------------------------------


def test_nis_loss_gradcheck() -> None:
    """torch.autograd.gradcheck on NIS loss w.r.t. L_Q and L_R through the KF."""

    def nis_from_raw(L_Q_raw: Tensor, L_R_raw: Tensor) -> Tensor:
        innov, inn_cov, states, state_covs = _run_kf(L_Q_raw, L_R_raw, _Z_SEQ)
        outputs = FilterOutputs(
            innovations=torch.stack(innov, dim=1),
            innovation_covs=torch.stack(inn_cov, dim=1),
            states=torch.stack(states, dim=1),
            state_covs=torch.stack(state_covs, dim=1),
        )
        return nis_consistency(outputs, lam=1.0)

    init = _SOFTPLUS_INV_1
    L_Q_raw = torch.eye(_DIM_X, dtype=torch.float64, requires_grad=True) * init
    L_R_raw = torch.eye(_DIM_Z, dtype=torch.float64, requires_grad=True) * init

    assert torch.autograd.gradcheck(nis_from_raw, (L_Q_raw, L_R_raw), eps=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# NEES gradcheck
# ---------------------------------------------------------------------------


def test_nees_loss_gradcheck() -> None:
    """torch.autograd.gradcheck on NEES loss w.r.t. L_Q and L_R through the KF."""

    def nees_from_raw(L_Q_raw: Tensor, L_R_raw: Tensor) -> Tensor:
        innov, inn_cov, states, state_covs = _run_kf(L_Q_raw, L_R_raw, _Z_SEQ)
        outputs = FilterOutputs(
            innovations=torch.stack(innov, dim=1),
            innovation_covs=torch.stack(inn_cov, dim=1),
            states=torch.stack(states, dim=1),
            state_covs=torch.stack(state_covs, dim=1),
            states_true=_X_TRUE,
        )
        return nees_consistency(outputs, lam=1.0)

    init = _SOFTPLUS_INV_1
    L_Q_raw = torch.eye(_DIM_X, dtype=torch.float64, requires_grad=True) * init
    L_R_raw = torch.eye(_DIM_Z, dtype=torch.float64, requires_grad=True) * init

    assert torch.autograd.gradcheck(nees_from_raw, (L_Q_raw, L_R_raw), eps=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# EKF shared fixtures
# ---------------------------------------------------------------------------

_EKF_DT = 0.05
_EKF_K = 1.0
_EKF_K3 = 0.5
_EKF_C = 0.2


def _duffing_f(x: Tensor, u: Tensor | None) -> Tensor:
    """Single-sample Duffing oscillator step (Euler, no control)."""
    pos, vel = x[0], x[1]
    return torch.stack([
        pos + vel * _EKF_DT,
        vel + (-_EKF_K * pos - _EKF_K3 * pos ** 3 - _EKF_C * vel) * _EKF_DT,
    ])


def _pos_h(x: Tensor) -> Tensor:
    """Position-only measurement."""
    return x[:1]


def _run_ekf(
    L_Q_raw: Tensor,
    L_R_raw: Tensor,
    z_seq: Tensor,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Run a T-step EKF (Duffing) and return (innovations, inn_covs, states, state_covs)."""
    ekf = ExtendedKalmanFilter(f=_duffing_f, h=_pos_h, dim_x=_DIM_X, dim_z=_DIM_Z)
    Q = _build_spd(L_Q_raw, _DIM_X)
    R = _build_spd(L_R_raw, _DIM_Z)

    B = z_seq.shape[0]
    x = torch.zeros(B, _DIM_X, dtype=L_Q_raw.dtype)
    P = torch.eye(_DIM_X, dtype=L_Q_raw.dtype).unsqueeze(0).expand(B, -1, -1).clone()

    innovations, inn_covs, states, state_covs = [], [], [], []
    for t in range(_T):
        x, P = ekf.predict(x, P, Q)
        x, P, y, S = ekf.update(x, P, z_seq[:, t, :], R)
        innovations.append(y)
        inn_covs.append(S)
        states.append(x)
        state_covs.append(P)

    return innovations, inn_covs, states, state_covs


# ---------------------------------------------------------------------------
# EKF reduces to KF when f and h are linear
# ---------------------------------------------------------------------------


def test_ekf_reduces_to_kf() -> None:
    """EKF with linear f and h must match KalmanFilter to 1e-8 over 30 steps."""
    Q = torch.tensor([[1e-3, 0.0], [0.0, 1e-2]], dtype=torch.float64)
    R = torch.tensor([[0.5]], dtype=torch.float64)

    def f_lin(x: Tensor, u: Tensor | None) -> Tensor:
        return _F @ x  # Jacobian = _F (constant)

    def h_lin(x: Tensor) -> Tensor:
        return _H @ x  # Jacobian = _H (constant)

    ekf = ExtendedKalmanFilter(f=f_lin, h=h_lin, dim_x=_DIM_X, dim_z=_DIM_Z)
    kf = KalmanFilter(_F, _H)

    B, T = 4, 30
    torch.manual_seed(3)
    z_seq = torch.randn(B, T, _DIM_Z, dtype=torch.float64) * 0.5

    x0 = torch.zeros(B, _DIM_X, dtype=torch.float64)
    P0 = torch.eye(_DIM_X, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).clone()
    x_e, P_e = x0.clone(), P0.clone()
    x_k, P_k = x0.clone(), P0.clone()

    with torch.no_grad():
        for t in range(T):
            z_t = z_seq[:, t, :]
            x_e, P_e = ekf.predict(x_e, P_e, Q)
            x_e, P_e, y_e, S_e = ekf.update(x_e, P_e, z_t, R)
            x_k, P_k = kf.predict(x_k, P_k, Q)
            x_k, P_k, y_k, S_k = kf.update(x_k, P_k, z_t, R)

            assert torch.allclose(x_e, x_k, atol=1e-8), f"x mismatch at step {t}"
            assert torch.allclose(P_e, P_k, atol=1e-8), f"P mismatch at step {t}"
            assert torch.allclose(y_e, y_k, atol=1e-8), f"y mismatch at step {t}"
            assert torch.allclose(S_e, S_k, atol=1e-8), f"S mismatch at step {t}"


# ---------------------------------------------------------------------------
# EKF NIS gradcheck
# ---------------------------------------------------------------------------


def test_ekf_nis_loss_gradcheck() -> None:
    """torch.autograd.gradcheck on NIS loss through the EKF (Duffing oscillator)."""

    def nis_from_raw(L_Q_raw: Tensor, L_R_raw: Tensor) -> Tensor:
        innov, inn_cov, states, state_covs = _run_ekf(L_Q_raw, L_R_raw, _Z_SEQ)
        outputs = FilterOutputs(
            innovations=torch.stack(innov, dim=1),
            innovation_covs=torch.stack(inn_cov, dim=1),
            states=torch.stack(states, dim=1),
            state_covs=torch.stack(state_covs, dim=1),
        )
        return nis_consistency(outputs, lam=1.0)

    init = _SOFTPLUS_INV_1
    L_Q_raw = torch.eye(_DIM_X, dtype=torch.float64, requires_grad=True) * init
    L_R_raw = torch.eye(_DIM_Z, dtype=torch.float64, requires_grad=True) * init

    assert torch.autograd.gradcheck(nis_from_raw, (L_Q_raw, L_R_raw), eps=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# EKF NEES gradcheck
# ---------------------------------------------------------------------------


def test_ekf_nees_loss_gradcheck() -> None:
    """torch.autograd.gradcheck on NEES loss through the EKF (Duffing oscillator)."""

    def nees_from_raw(L_Q_raw: Tensor, L_R_raw: Tensor) -> Tensor:
        innov, inn_cov, states, state_covs = _run_ekf(L_Q_raw, L_R_raw, _Z_SEQ)
        outputs = FilterOutputs(
            innovations=torch.stack(innov, dim=1),
            innovation_covs=torch.stack(inn_cov, dim=1),
            states=torch.stack(states, dim=1),
            state_covs=torch.stack(state_covs, dim=1),
            states_true=_X_TRUE,
        )
        return nees_consistency(outputs, lam=1.0)

    init = _SOFTPLUS_INV_1
    L_Q_raw = torch.eye(_DIM_X, dtype=torch.float64, requires_grad=True) * init
    L_R_raw = torch.eye(_DIM_Z, dtype=torch.float64, requires_grad=True) * init

    assert torch.autograd.gradcheck(nees_from_raw, (L_Q_raw, L_R_raw), eps=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# UKF shared fixtures
# ---------------------------------------------------------------------------


def _run_ukf(
    L_Q_raw: Tensor,
    L_R_raw: Tensor,
    z_seq: Tensor,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Run a T-step UKF (Duffing) and return (innovations, inn_covs, states, state_covs).

    Uses alpha=0.3 to avoid the extreme weight magnitudes that arise with the
    default alpha=1e-3 (W_m[0] ≈ -999999), which would destabilise gradcheck.
    """
    ukf = UnscentedKalmanFilter(
        f=_duffing_f, h=_pos_h, dim_x=_DIM_X, dim_z=_DIM_Z, alpha=0.3
    )
    Q = _build_spd(L_Q_raw, _DIM_X)
    R = _build_spd(L_R_raw, _DIM_Z)

    B = z_seq.shape[0]
    x = torch.zeros(B, _DIM_X, dtype=L_Q_raw.dtype)
    P = torch.eye(_DIM_X, dtype=L_Q_raw.dtype).unsqueeze(0).expand(B, -1, -1).clone()

    innovations, inn_covs, states, state_covs = [], [], [], []
    for t in range(_T):
        x, P = ukf.predict(x, P, Q)
        x, P, y, S = ukf.update(x, P, z_seq[:, t, :], R)
        innovations.append(y)
        inn_covs.append(S)
        states.append(x)
        state_covs.append(P)

    return innovations, inn_covs, states, state_covs


# ---------------------------------------------------------------------------
# UKF reduces to KF when f and h are linear
# ---------------------------------------------------------------------------


def test_ukf_reduces_to_kf() -> None:
    """UKF with linear f and h must match KalmanFilter to 1e-6 over 30 steps."""
    Q = torch.tensor([[1e-3, 0.0], [0.0, 1e-2]], dtype=torch.float64)
    R = torch.tensor([[0.5]], dtype=torch.float64)

    def f_lin(x: Tensor, u: Tensor | None) -> Tensor:
        return _F @ x

    def h_lin(x: Tensor) -> Tensor:
        return _H @ x

    # alpha=0.3 gives moderate weights (W_m[0] ≈ -10) rather than the extreme
    # alpha=1e-3 weights (W_m[0] ≈ -999999); algebraically both equal KF for
    # linear f/h, but extreme weights accumulate floating-point cancellation.
    ukf = UnscentedKalmanFilter(f=f_lin, h=h_lin, dim_x=_DIM_X, dim_z=_DIM_Z, alpha=0.3)
    kf = KalmanFilter(_F, _H)

    B, T = 4, 30
    torch.manual_seed(5)
    z_seq = torch.randn(B, T, _DIM_Z, dtype=torch.float64) * 0.5

    x0 = torch.zeros(B, _DIM_X, dtype=torch.float64)
    P0 = torch.eye(_DIM_X, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).clone()
    x_u, P_u = x0.clone(), P0.clone()
    x_k, P_k = x0.clone(), P0.clone()

    with torch.no_grad():
        for t in range(T):
            z_t = z_seq[:, t, :]
            x_u, P_u = ukf.predict(x_u, P_u, Q)
            x_u, P_u, y_u, S_u = ukf.update(x_u, P_u, z_t, R)
            x_k, P_k = kf.predict(x_k, P_k, Q)
            x_k, P_k, y_k, S_k = kf.update(x_k, P_k, z_t, R)

            assert torch.allclose(x_u, x_k, atol=1e-6), f"x mismatch at step {t}"
            assert torch.allclose(P_u, P_k, atol=1e-6), f"P mismatch at step {t}"
            assert torch.allclose(y_u, y_k, atol=1e-6), f"y mismatch at step {t}"
            assert torch.allclose(S_u, S_k, atol=1e-6), f"S mismatch at step {t}"


# ---------------------------------------------------------------------------
# UKF NIS gradcheck
# ---------------------------------------------------------------------------


def test_ukf_nis_loss_gradcheck() -> None:
    """torch.autograd.gradcheck on NIS loss through the UKF (Duffing oscillator)."""

    def nis_from_raw(L_Q_raw: Tensor, L_R_raw: Tensor) -> Tensor:
        innov, inn_cov, states, state_covs = _run_ukf(L_Q_raw, L_R_raw, _Z_SEQ)
        outputs = FilterOutputs(
            innovations=torch.stack(innov, dim=1),
            innovation_covs=torch.stack(inn_cov, dim=1),
            states=torch.stack(states, dim=1),
            state_covs=torch.stack(state_covs, dim=1),
        )
        return nis_consistency(outputs, lam=1.0)

    init = _SOFTPLUS_INV_1
    L_Q_raw = torch.eye(_DIM_X, dtype=torch.float64, requires_grad=True) * init
    L_R_raw = torch.eye(_DIM_Z, dtype=torch.float64, requires_grad=True) * init

    assert torch.autograd.gradcheck(nis_from_raw, (L_Q_raw, L_R_raw), eps=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# UKF NEES gradcheck
# ---------------------------------------------------------------------------


def test_ukf_nees_loss_gradcheck() -> None:
    """torch.autograd.gradcheck on NEES loss through the UKF (Duffing oscillator)."""

    def nees_from_raw(L_Q_raw: Tensor, L_R_raw: Tensor) -> Tensor:
        innov, inn_cov, states, state_covs = _run_ukf(L_Q_raw, L_R_raw, _Z_SEQ)
        outputs = FilterOutputs(
            innovations=torch.stack(innov, dim=1),
            innovation_covs=torch.stack(inn_cov, dim=1),
            states=torch.stack(states, dim=1),
            state_covs=torch.stack(state_covs, dim=1),
            states_true=_X_TRUE,
        )
        return nees_consistency(outputs, lam=1.0)

    init = _SOFTPLUS_INV_1
    L_Q_raw = torch.eye(_DIM_X, dtype=torch.float64, requires_grad=True) * init
    L_R_raw = torch.eye(_DIM_Z, dtype=torch.float64, requires_grad=True) * init

    assert torch.autograd.gradcheck(nees_from_raw, (L_Q_raw, L_R_raw), eps=1e-5, rtol=1e-3)
