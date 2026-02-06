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
    savepath: str | None = None,
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
    savepath : str, optional
        If set, save the figure to this path with bbox_inches="tight".

    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    x = np.asarray(result["x"])
    t = np.asarray(result["time"])
    c = np.asarray(result["c"])
    # Position in µm for display
    x_um = x / 1e-6

    if ax is None:
        _, ax = plt.subplots()

    n_times = c.shape[0]
    if time_idx is None:
        indices = list(range(n_times))
    elif isinstance(time_idx, int):
        indices = [time_idx]
    else:
        indices = list(time_idx)

    t_plot = np.array([t[k] for k in indices if 0 <= k < n_times])
    norm = Normalize(vmin=t_plot.min(), vmax=t_plot.max())
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    for k in indices:
        if 0 <= k < n_times:
            color = sm.cmap(norm(t[k]))
            ax.plot(x_um, c[k], color=color)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("concentration (m⁻³)")
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("time (s)")
    ax.grid(True, alpha=0.3)
    ax.get_figure().subplots_adjust(left=0.22)
    if savepath:
        ax.get_figure().savefig(savepath, bbox_inches="tight")
    return ax


def plot_fluxes(
    result: dict[str, Any],
    ax: Any = None,
    savepath: str | None = None,
) -> Any:
    """
    Plot inlet (desorbed) and outlet (permeation) flux vs time.

    Parameters
    ----------
    result : dict
        Solver output from BE() with key "fluxes" (DataFrame with time, rel, perm).
    ax : matplotlib axes, optional
        Axes to plot on; if None, use current axes (gca).
    savepath : str, optional
        If set, save the figure to this path with bbox_inches="tight".

    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt

    fluxes = result["fluxes"]
    if ax is None:
        _, ax = plt.subplots()

    t = fluxes["time"]
    ax.plot(t, fluxes["rel"], label="inlet (desorbed)", color="C0")
    ax.plot(t, fluxes["perm"], label="outlet (permeation)", color="C1")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("flux (m⁻² s⁻¹)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.get_figure().subplots_adjust(left=0.22)
    if savepath:
        ax.get_figure().savefig(savepath, bbox_inches="tight")
    return ax
