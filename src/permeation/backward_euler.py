"""
Backward Euler solver for 1D permeation PDE with recombination boundary conditions.

Spatial discretisation: finite differences. Boundary concentrations from quadratic
BCs; iterative correction (linearise u² as u·a with a from previous iterate).
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def BE(
    Nx: int = 30,
    Nt: int = 100,
    T: float = 100.0,
    D: float = 1.1e-8,
    L: float = 2e-5,
    ku: float = 1e-33,
    kd: float = 1e-33,
    ks: float = 1e16,
    G: np.ndarray | None = None,
    I: Callable[[float], float] | None = None,
    Uinit: np.ndarray | None = None,
    PLOT: bool = False,
    saveU: str | None = None,
    ncorrection: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Backward Euler step for permeation: ∂u/∂t = D ∂²u/∂x² with flux/recombination BCs.

    Boundary conditions (upstream x=0, downstream x=L):
    Γ_in + D ∂u/∂x|_0 - k_u u(0)² = 0,  -D ∂u/∂x|_L - k_d u(L)² = 0.

    Parameters
    ----------
    Nx, Nt : int
        Space and time mesh points (interior + boundaries).
    T : float
        End time [s].
    D : float
        Diffusion coefficient [m²/s].
    L : float
        Membrane thickness [m].
    ku, kd : float
        Upstream and downstream recombination coefficients.
    ks : float
        Scale for incident flux (G is multiplied by ks).
    G : array of shape (Nt+1,)
        Incident flux (dimensionless before scaling).
    I : callable, optional
        Initial concentration profile u(x); if None, Uinit is used.
    Uinit : array of shape (Nx+1,), optional
        Initial concentration; used if I is None.
    PLOT : bool
        If True, matplotlib is used to plot concentration evolution (not in API guarantee).
    saveU, ncorrection : optional
        saveU unused in this implementation; ncorrection = number of BC correction iterations.

    Returns
    -------
    dict
        "fluxes" : DataFrame with columns time, rel (inlet), perm (outlet)
        "c" : (Nt+1, Nx+1) concentration history
        "calctime" : float, seconds
    """
    # Platform-specific factor for TMAP7 agreement (H2 -> 2H interpretation)
    if os.name == "nt":
        ku = 2.0 * ku
        kd = 2.0 * kd

    if G is None:
        G = np.zeros(Nt + 1)
    G = G * ks
    if np.any(G < 0):
        return {
            "time": np.linspace(0, T, Nt + 1),
            "pdp": np.zeros(Nt + 1),
            "fluxes": pd.DataFrame({"time": np.linspace(0, T, Nt + 1), "rel": 0.0, "perm": 0.0}),
            "c": np.zeros((Nt + 1, Nx + 1)),
            "calctime": 0.0,
        }

    t0 = time.perf_counter()
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T, Nt + 1)
    if I is not None:
        u_1 = np.array([I(xi) for xi in x], dtype=float)
    else:
        u_1 = np.copy(Uinit if Uinit is not None else np.zeros(Nx + 1))

    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    F = D * dt / (dx**2)
    teta1 = D / (ku * dx)
    teta2 = D / (kd * dx)

    inlet: list[float] = [float(ku * u_1[0] ** 2)]
    outlet: list[float] = [float(kd * u_1[Nx] ** 2)]
    u = np.zeros(Nx + 1)
    Usave = np.zeros((Nt + 1, Nx + 1), dtype=float)
    Usave[0] = u_1

    if PLOT:
        import matplotlib.pyplot as plt
        plt.plot(x / 1e-6, u_1, "k-", lw=4)
    color_idx = np.linspace(0, 1, Nt)

    for n in range(Nt):
        # Explicit guess for boundary neighbours (equations 3.2, 3.3)
        g0 = F * u_1[0] + (1 - 2 * F) * u_1[1] + F * u_1[2]
        gL = F * u_1[Nx - 2] + (1 - 2 * F) * u_1[Nx - 1] + F * u_1[Nx]
        # Initial A with Neumann-style boundary rows (1 on diagonal)
        A = diags(
            diagonals=[
                [0.0] + [-F] * (Nx - 1),
                [1.0] + [1.0 + 2.0 * F] * (Nx - 1) + [1.0],
                [-F] * (Nx - 1) + [0.0],
            ],
            offsets=[1, 0, -1],
            shape=(Nx + 1, Nx + 1),
            format="csr",
        )
        # RHS: quadratic roots for boundaries, interior = u_1
        b = np.concatenate([
            [-teta1 / 2.0 + 0.5 * np.sqrt(teta1**2 + 4 * teta1 * g0 + 4 * G[n] / ku)],
            u_1[1:Nx],
            [-teta2 / 2.0 + 0.5 * np.sqrt(teta2**2 + 4 * teta2 * gL)],
        ])
        u[:] = spsolve(A, b)

        # Iterative correction: linearise u² as u·a (a = previous u)
        for _ in range(ncorrection):
            a0, aL = u[0], u[Nx]
            A = diags(
                diagonals=[
                    [-D / dx] + [-F] * (Nx - 1),
                    [D / dx + ku * a0] + [1.0 + 2.0 * F] * (Nx - 1) + [D / dx + kd * aL],
                    [-F] * (Nx - 1) + [-D / dx],
                ],
                offsets=[1, 0, -1],
                shape=(Nx + 1, Nx + 1),
                format="csr",
            )
            b = np.concatenate([[G[n]], u_1[1:Nx], [0.0]])
            u[:] = spsolve(A, b)

        u_1, u = u, u_1
        inlet.append(float(ku * u_1[0] ** 2))
        outlet.append(float(kd * u_1[Nx] ** 2))
        if PLOT:
            import matplotlib.pyplot as plt
            plt.plot(x / 1e-6, u_1, ".-", color=plt.cm.jet(float(color_idx[n])))
        Usave[n + 1] = u_1

    if PLOT:
        import matplotlib.pyplot as plt
        font = {"family": "serif", "weight": "normal", "size": 12}
        plt.rc("font", **font)
        plt.rcParams.update({"mathtext.default": "regular"})
        ax = plt.gca()
        ax.set_xlim(0, L / 1e-6)
        ax.set_xlabel(r"x ($\mu$m)")
        ax.set_ylabel("concentration (m$^{-3}$)")

    elapsed = time.perf_counter() - t0
    fluxes_df = pd.DataFrame({"time": t, "rel": inlet, "perm": outlet})
    return {
        "fluxes": fluxes_df,
        "c": Usave,
        "calctime": elapsed,
        "time": t,
        "pdp": np.array(outlet),
    }
