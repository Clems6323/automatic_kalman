"""Automatic Kalman — learn Q and R by gradient descent."""

from automatic_kalman.covariance import CholeskyCovariance
from automatic_kalman.losses import nees_consistency, nis_consistency
from automatic_kalman.train import train
from automatic_kalman.types import FilterOutputs

__all__ = [
    "CholeskyCovariance",
    "FilterOutputs",
    "nis_consistency",
    "nees_consistency",
    "train",
]
