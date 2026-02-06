"""
Plotting helpers for permeation solver output.

Operate only on the result dictionary returned by BE(). No solver internals
or global state. Requires matplotlib.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def plot_profiles(
    result: dict[str, Any],
    time_idx: int | Sequence[int] | None = None,
    ax: Any = None,
) -> Any:
    """
    Plot concentration vs position (spatial profiles).

    Parameters
    ----------
    result : dict
        Solver output from BE() with keys "x", "time", "c".
    time_idx : int, sequence of int, or None, optional
        Time indices to plot. None = all time steps.
    ax : matplotlib axes, optional
        Axes to plot on; if None, use current axes (gca).

    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt

    x = np.asarray(result["x"])
    t = np.asarray(result["time"])
    c = np.asarray(result["c"])
    # Position in µm for display
    x_um = x / 1e-6

    if ax is None:
        ax = plt.gca()

    n_times = c.shape[0]
    if time_idx is None:
        indices = list(range(n_times))
    elif isinstance(time_idx, int):
        indices = [time_idx]
    else:
        indices = list(time_idx)

    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(indices):
        if 0 <= k < n_times:
            color = cmap(i / max(len(indices), 1))
            ax.plot(x_um, c[k], color=color, label=f"t = {t[k]:.2f} s")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("concentration (m⁻³)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)
    return ax


def plot_fluxes(result: dict[str, Any], ax: Any = None) -> Any:
    """
    Plot inlet (reflected) and outlet (permeation) flux vs time.

    Parameters
    ----------
    result : dict
        Solver output from BE() with key "fluxes" (DataFrame with time, rel, perm).
    ax : matplotlib axes, optional
        Axes to plot on; if None, use current axes (gca).

    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt

    fluxes = result["fluxes"]
    if ax is None:
        ax = plt.gca()

    t = fluxes["time"]
    ax.plot(t, fluxes["rel"], label="inlet (reflected)", color="C0")
    ax.plot(t, fluxes["perm"], label="outlet (permeation)", color="C1")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("flux (m⁻² s⁻¹)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax
