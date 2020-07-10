from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
import numpy as np
import time
import matplotlib.pylab as plt
import os

from numba import jit


def BE(**kwargs):
    """Calculate Permeated Flux"""
    Nx = kwargs.get("Nx", 30)
    Nt = kwargs.get("Nt", 100)
    T = kwargs.get("T", 100)  # 0-T - time interval for calculation
    D = kwargs.get("D", 1.1e-8)
    I = kwargs.get("I", None)
    G = kwargs.get("G", None)
    ku = kwargs.get("ku", 1e-33)
    kd = kwargs.get("kd", 1e-33)
    ks = kwargs.get("ks", 1e16)
    L = kwargs.get("L", 2e-5)
    PLOT = kwargs.get("PLOT", False)
    saveU = kwargs.get("saveU", None)
    Uinit = kwargs.get("Uinit", np.zeros(Nx + 1))
    # ------------------------------------------------------------------------------
    ku = 2 * ku
    kd = 2 * kd  # if this is applied, result is same with TMAP7. Why?
    # ------------------------------------------------------------------------------
    G = G * ks
    if len(np.where(G < 0)[0]):
        return {"time": np.linspace(0, T, Nt + 1), "pdp": np.zeros(Nt + 1)}
    start = time.clock()
    x = np.linspace(0, L, Nx + 1)  # mesh points in space
    t = np.linspace(0, T, Nt + 1)  # mesh points in time
    if I:
        u_1 = np.array([I(i) for i in x])  # initial concentration
    else:
        u_1 = np.copy(Uinit)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    F = D * dt / dx ** 2
    inlet = []
    outlet = []
    inlet.append(ku * u_1[0] ** 2)
    outlet.append(kd * u_1[Nx] ** 2)
    u = np.zeros(Nx + 1)
    if saveU:
        Usave = np.zeros((Nt + 1, Nx + 1), float)
    if saveU:
        Usave[0] = u_1
    if PLOT:
        plt.plot(x / 1e-6, u_1, "k-", lw=4)
    color_idx = np.linspace(0, 1, Nt)
    teta1 = D / (ku * dx)
    teta2 = D / (kd * dx)

    for n in range(0, Nt):
        # calculate u[1] and u[Nx-1] using explicit stencil
        g0 = F * u_1[0] + (1 - 2 * F) * u_1[1] + F * u_1[2]
        gL = F * u_1[Nx - 2] + (1 - 2 * F) * u_1[Nx - 1] + F * u_1[Nx]
        # put 1 for u[0] and u[Nx-1] in A
        A = diags(
            diagonals=[
                [0] + [-F for i in range(Nx - 1)],
                [1] + [1.0 + 2.0 * F for i in range(Nx - 1)] + [1],
                [-F for i in range(Nx - 1)] + [0],
            ],
            offsets=[1, 0, -1],
            shape=(Nx + 1, Nx + 1),
            format="csr",
        )
        # in the b (for BE) put roots of the quadratic equation for the border.
        b = np.array(
            [-teta1 / 2.0 + 0.5 * np.sqrt(teta1 ** 2 + 4 * teta1 * g0 + 4 * G[n] / ku)]
            + [i for i in u_1[1:Nx]]
            + [-teta2 / 2.0 + 0.5 * np.sqrt(teta2 ** 2 + 4 * teta2 * gL)]
        )
        # solve SLE
        u[:] = spsolve(A, b)
        u"Instead of u**2 put u*a, where a - u from previous step"
        for _ in range(3):
            a0 = u[0]
            aL = u[Nx]
            A = diags(
                diagonals=[
                    [-D / dx] + [-F for i in range(Nx - 1)],
                    [D / dx + ku * a0]
                    + [1.0 + 2.0 * F for i in range(Nx - 1)]
                    + [D / dx + kd * aL],
                    [-F for i in range(Nx - 1)] + [-D / dx],
                ],
                offsets=[1, 0, -1],
                shape=(Nx + 1, Nx + 1),
                format="csr",
            )
            b = np.array([G[n]] + [i for i in u_1[1:Nx]] + [0])
            u[:] = spsolve(A, b)
        u_1, u = u, u_1
        inlet.append(ku * u_1[0] ** 2)
        outlet.append(kd * u_1[Nx] ** 2)
        if PLOT:
            plt.plot(
                x / 1e-6, u_1, ".-", color=plt.cm.jet(color_idx[n])
            )  # @UndefinedVariable
        if saveU:
            Usave[n + 1] = u_1

    if PLOT:
        font = {"family": "Times New Roman", "weight": "heavy", "size": 25}
        plt.rc("font", **font)
        plt.rcParams.update({"mathtext.default": "regular"})
        ax = plt.gca()
        ax.set_xlim(0, L / 1e-6)
        ax.set_xlabel("x ($\mu m$)", fontweight="heavy")
        ax.set_ylabel("concentration ($m^{-3}$)", fontweight="heavy")

    end = time.clock()
    result = dict()
    result.update(
        [
            ("reflected", inlet),
            ("pdp", outlet),
            ("time", t),
            ("concentration", u_1),
            ("calctime", end - start),
        ]
    )
    if saveU:
        return Usave, [t, outlet]
    else:
        return result


def parameters():
    """initial parameters for the fit

    Nx - number of spacial points, Nt - number of time steps, T - end time for simulation,
    Tend - end of incident flux, D - diffusion coefficient, L - membrane thickness,
    I - probably initial concentraion in the membrane, Uinit - initial concentration in the membrane,
    G - incident flux,
    ku,kd - upstream and downstream recombination coefficients,
    ks - absolute intensity multiplier for incident flux,
    PLOT - show plot or not after calculation.
    """
    params = {
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
    }

    params["G"] = np.zeros(params["Nt"] + 1)
    params["Uinit"] = np.zeros(params["Nx"] + 1)

    return params

