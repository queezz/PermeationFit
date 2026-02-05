"""Utility functions for permeation (e.g. fitting metrics)."""

from __future__ import annotations

import numpy as np


def chi_square(exp: np.ndarray | list[float], calc: np.ndarray | list[float]) -> float:
    """
    Normalised sum of squared differences between two time series.

    Uses max(exp) as scale so result is dimensionless. Useful as a cost for fitting
    calculated permeation flux to experimental data.
    """
    exp_arr = np.asarray(exp)
    calc_arr = np.asarray(calc)
    M = float(np.max(exp_arr))
    if M == 0:
        M = 1.0
    return float(np.sum((exp_arr - calc_arr) ** 2 / M**2))
