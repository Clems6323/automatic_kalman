"""KF vs NumPy baseline parity tests — step-for-step to 1e-10 in float64."""

import numpy as np
import torch

from automatic_kalman.filters.kf import KalmanFilter
from baselines.kf_numpy import KalmanFilterNumpy


def _cv_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constant-velocity: 2 states (pos, vel), 1 measurement (pos), dt=0.1."""
    dt = 0.1
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1e-3, 0.0], [0.0, 1e-2]])
    R = np.array([[0.5]])
    return F, H, Q, R


def _fixed_measurements(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    T: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Generate a fixed deterministic sequence of measurements."""
    rng = np.random.default_rng(seed)
    dim_x = F.shape[0]
    dim_z = H.shape[0]
    L_Q = np.linalg.cholesky(Q)
    L_R = np.linalg.cholesky(R)
    x = np.array([0.0, 1.0])
    measurements = []
    for _ in range(T):
        x = F @ x + L_Q @ rng.standard_normal(dim_x)
        z = H @ x + L_R @ rng.standard_normal(dim_z)
        measurements.append(z.copy())
    return np.array(measurements)  # (T, dim_z)


def test_kf_matches_numpy_baseline() -> None:
    """PyTorch KF must match NumPy KalmanFilterNumpy to 1e-10 in float64 over 60 steps.

    Compares x_pred, P_pred, x_post, P_post, innovation y, and innovation cov S
    at every step.  Both implementations use Joseph-form covariance update and
    symmetrisation, so 1e-10 agreement is expected in float64.
    """
    F_np, H_np, Q_np, R_np = _cv_system()
    T = 60
    dim_x = F_np.shape[0]
    measurements = _fixed_measurements(F_np, H_np, Q_np, R_np, T=T)

    x0_np = np.zeros(dim_x)
    P0_np = np.eye(dim_x) * 10.0

    kf_np = KalmanFilterNumpy(F_np, H_np)
    kf_t = KalmanFilter(F=torch.from_numpy(F_np), H=torch.from_numpy(H_np))

    Q_t = torch.from_numpy(Q_np)
    R_t = torch.from_numpy(R_np)
    x_np, P_np = x0_np.copy(), P0_np.copy()
    x_t = torch.from_numpy(x0_np).unsqueeze(0)   # (1, dim_x)
    P_t = torch.from_numpy(P0_np).unsqueeze(0)   # (1, dim_x, dim_x)

    atol = 1e-10

    for k in range(T):
        z_np = measurements[k]
        z_t = torch.from_numpy(z_np).unsqueeze(0)  # (1, dim_z)

        # ── predict ──────────────────────────────────────────────────────────
        x_pred_np, P_pred_np = kf_np.predict(x_np, P_np, Q_np)
        x_pred_t, P_pred_t = kf_t.predict(x_t, P_t, Q_t)

        np.testing.assert_allclose(
            x_pred_t.squeeze(0).detach().numpy(), x_pred_np, atol=atol,
            err_msg=f"x_pred mismatch at step {k}",
        )
        np.testing.assert_allclose(
            P_pred_t.squeeze(0).detach().numpy(), P_pred_np, atol=atol,
            err_msg=f"P_pred mismatch at step {k}",
        )

        # ── update ───────────────────────────────────────────────────────────
        x_post_np, P_post_np, y_np, S_np = kf_np.update(x_pred_np, P_pred_np, z_np, R_np)
        x_post_t, P_post_t, y_t, S_t = kf_t.update(x_pred_t, P_pred_t, z_t, R_t)

        np.testing.assert_allclose(
            x_post_t.squeeze(0).detach().numpy(), x_post_np, atol=atol,
            err_msg=f"x_post mismatch at step {k}",
        )
        np.testing.assert_allclose(
            P_post_t.squeeze(0).detach().numpy(), P_post_np, atol=atol,
            err_msg=f"P_post mismatch at step {k}",
        )
        np.testing.assert_allclose(
            y_t.squeeze(0).detach().numpy(), y_np, atol=atol,
            err_msg=f"innovation mismatch at step {k}",
        )
        np.testing.assert_allclose(
            S_t.squeeze(0).detach().numpy(), S_np, atol=atol,
            err_msg=f"S mismatch at step {k}",
        )

        x_np, P_np = x_post_np, P_post_np
        x_t, P_t = x_post_t, P_post_t
