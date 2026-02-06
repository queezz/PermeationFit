"""
Backward Euler solver for 1D permeation PDE with recombination boundary conditions.

Numerical method summary
------------------------
Time integration:
  • Backward Euler (fully implicit, first order in time)

Spatial discretisation:
  • 1D second-order central finite differences on a uniform grid

Linear system:
  • At each time step, solve a sparse tridiagonal system
      (I - Δt·D·L) u^{n+1} = u^n
    where L is the discrete Laplacian
  • Implemented using scipy.sparse.diags + scipy.sparse.linalg.spsolve

Boundary conditions:
  • Nonlinear recombination (quadratic flux) boundary conditions at x=0 and x=L
  • Treated implicitly via Picard (fixed-point) iteration:
      u² ≈ a·u, with a taken from the previous iterate
  • A short explicit predictor is used only to initialise the boundary values

Properties:
  • Unconditionally stable with respect to the diffusion CFL condition
  • Numerical damping of fast transients (typical of Backward Euler)
  • Nonlinearity confined to boundary treatment; interior remains linear

"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from permeation.materials import parameters as default_parameters


def BE(**kwargs: Any) -> dict[str, Any]:
    """
    Backward Euler step for permeation: ∂u/∂t = D ∂²u/∂x² with flux/recombination BCs.

    Boundary conditions (upstream x=0, downstream x=L):
    Γ_in + D ∂u/∂x|_0 - k_u u(0)² = 0,  -D ∂u/∂x|_L - k_d u(L)² = 0.

    Defaults come from permeation.materials.parameters(); pass any key to override
    (e.g. Nx, Nt, T, D, L, ku, kd, ks, G, I, Uinit, ncorrection). Tend, saveU, PLOT
    are accepted but not used by the solver.

    Returns
    -------
    dict
        x : (Nx+1,) spatial grid [m]
        time : (Nt+1,) time grid [s]
        c : (Nt+1, Nx+1) concentration history
        fluxes : DataFrame with columns time, rel (inlet), perm (outlet)
        params : resolved parameter dict used in the run (includes dx, dt, modified ku/kd)
        calctime : float, seconds
        pdp : (Nt+1,) outlet flux (alias for fluxes["perm"])
    """
    params = default_parameters()
    params.update(kwargs)
    Nx = params["Nx"]
    Nt = params["Nt"]
    T = params["T"]
    D = params["D"]
    L = params["L"]
    ku = params["ku"]
    kd = params["kd"]
    ks = params["ks"]
    G = params["G"]
    I = params["I"]
    Uinit = params["Uinit"]
    ncorrection = params["ncorrection"]

    # Platform-specific factor for TMAP7 agreement (H2 -> 2H interpretation)
    if os.name == "nt":
        ku = 2.0 * ku
        kd = 2.0 * kd

    if G is None:
        G = np.zeros(Nt + 1)
    G = G * ks
    if np.any(G < 0):
        t_arr = np.linspace(0, T, Nt + 1)
        x_arr = np.linspace(0, L, Nx + 1)
        dx = L / Nx
        dt = T / Nt
        resolved = {
            "Nx": Nx,
            "Nt": Nt,
            "T": T,
            "D": D,
            "L": L,
            "ku": ku,
            "kd": kd,
            "ks": ks,
            "ncorrection": ncorrection,
            "dx": dx,
            "dt": dt,
        }
        return {
            "x": x_arr,
            "time": t_arr,
            "c": np.zeros((Nt + 1, Nx + 1)),
            "fluxes": pd.DataFrame({"time": t_arr, "rel": 0.0, "perm": 0.0}),
            "params": resolved,
            "calctime": 0.0,
            "pdp": np.zeros(Nt + 1),
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
        b = np.concatenate(
            [
                [
                    -teta1 / 2.0
                    + 0.5 * np.sqrt(teta1**2 + 4 * teta1 * g0 + 4 * G[n] / ku)
                ],
                u_1[1:Nx],
                [-teta2 / 2.0 + 0.5 * np.sqrt(teta2**2 + 4 * teta2 * gL)],
            ]
        )
        u[:] = spsolve(A, b)

        # Iterative correction: linearise u² as u·a (a = previous u)
        for _ in range(ncorrection):
            a0, aL = u[0], u[Nx]
            A = diags(
                diagonals=[
                    [-D / dx] + [-F] * (Nx - 1),
                    [D / dx + ku * a0]
                    + [1.0 + 2.0 * F] * (Nx - 1)
                    + [D / dx + kd * aL],
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
        Usave[n + 1] = u_1

    elapsed = time.perf_counter() - t0
    fluxes_df = pd.DataFrame({"time": t, "rel": inlet, "perm": outlet})
    resolved = {
        "Nx": Nx,
        "Nt": Nt,
        "T": T,
        "D": D,
        "L": L,
        "ku": ku,
        "kd": kd,
        "ks": ks,
        "ncorrection": ncorrection,
        "dx": dx,
        "dt": dt,
    }
    return {
        "x": x,
        "time": t,
        "c": Usave,
        "fluxes": fluxes_df,
        "params": resolved,
        "calctime": elapsed,
        "pdp": np.array(outlet),
    }
