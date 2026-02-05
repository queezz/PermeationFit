"""Default material and problem parameters for permeation simulations."""

from __future__ import annotations

import numpy as np
from typing import Any


def parameters() -> dict[str, Any]:
    """
    Default parameters for the permeation solver.

    Returns
    -------
    dict
        Nx, Nt, T, D, Tend, L, I, G, Uinit, ku, kd, ks, PLOT, ncorrection.
    """
    params: dict[str, Any] = {
        "Nx": 30,
        "Nt": 100,
        "T": 1000.0,
        "D": 1.1e-8,
        "Tend": 705.0,
        "L": 2e-5,
        "I": None,
        "ku": 1e-33,
        "kd": 2e-33,
        "ks": 1e19,
        "PLOT": False,
        "ncorrection": 3,
    }
    params["G"] = np.zeros(params["Nt"] + 1)
    params["Uinit"] = np.zeros(params["Nx"] + 1)
    return params
