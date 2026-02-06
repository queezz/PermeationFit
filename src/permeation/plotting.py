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


def plot_concentration_3d(
    result: dict[str, Any],
    ax: Any = None,
    cmap: str = "inferno",
    figsize: Sequence[float] = (12, 8),
    savepath: str | None = None,
) -> Any:
    """
    Plot concentration profile time evolution as a 3D surface.

    Surface: x = position (µm), y = time (s), z = concentration scaled by 10^pwr
    so the z-axis label is "c, 10^{pwr} H/m³".

    Parameters
    ----------
    result : dict
        Solver output from BE() with keys "x", "time", "c".
    ax : matplotlib 3D axes, optional
        Axes to plot on; if None, create a new figure with 3D projection.
    cmap : str, optional
        Colormap name for the surface (default "inferno").
    figsize : sequence of float, optional
        (width, height) in inches when creating a new figure (default (12, 8)).
    savepath : str, optional
        If set, save the figure to this path with bbox_inches="tight".

    Returns
    -------
    matplotlib 3D axes
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm

    x = np.asarray(result["x"])
    t = np.asarray(result["time"])
    c = np.asarray(result["c"], dtype=float)
    x_um = x / 1e-6
    X, Y = np.meshgrid(x_um, t)

    c_max = float(np.max(c))
    pwr = int(np.round(np.log10(c_max))) if c_max > 0 else 0
    scale = 10.0 ** pwr
    a = c / scale if scale > 0 else c

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    cmap_obj = mpl_cm.get_cmap(cmap)
    surf = ax.plot_surface(
        X, Y, a,
        cmap=cmap_obj,
        vmin=0.0,
        vmax=np.max(a) if a.size else 1.0,
        edgecolor="C2",
        linewidth=0.2,
    )
    surf.set_facecolor((0, 0, 0, 0))
    ax.get_figure().colorbar(surf, shrink=0.6, aspect=10)
    ax.view_init(25, 40)
    ax.set_xlabel("d (µm)", labelpad=15)
    ax.set_ylabel("time (s)", labelpad=20)
    ax.set_zlabel(f"c, $10^{{{pwr}}}$ H/m³", labelpad=15)
    if savepath:
        ax.get_figure().savefig(savepath, bbox_inches="tight")
    return ax


def plot_summary(
    result: dict[str, Any],
    profile_time_idx: int | None = None,
    cmap: str = "inferno",
    figsize: Sequence[float] = (14, 8),
    savepath: str | None = None,
) -> tuple[Any, Any, Any]:
    """
    Plot fluxes, a single (flat) concentration profile, and 3D surface in one figure.

    Parameters
    ----------
    result : dict
        Solver output from BE() with keys "x", "time", "c", "fluxes".
    profile_time_idx : int or None, optional
        Time index for the flat profile. None = last available time step.
    cmap : str, optional
        Colormap name for the 3D surface (default "inferno").
    figsize : sequence of float, optional
        (width, height) in inches for the combined figure.
    savepath : str, optional
        If set, save the figure to this path with bbox_inches="tight".

    Returns
    -------
    tuple of matplotlib axes
        (ax_fluxes, ax_profile, ax_3d)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, width_ratios=(1.0, 1.2))
    ax_fluxes = fig.add_subplot(gs[0, 0])
    ax_profile = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    plot_fluxes(result, ax=ax_fluxes)

    x = np.asarray(result["x"])
    t = np.asarray(result["time"])
    c = np.asarray(result["c"])
    x_um = x / 1e-6

    if c.size == 0 or c.ndim < 2:
        ax_profile.text(
            0.5,
            0.5,
            "no concentration data",
            ha="center",
            va="center",
            transform=ax_profile.transAxes,
        )
    else:
        n_times = c.shape[0]
        idx = n_times - 1 if profile_time_idx is None else int(profile_time_idx)
        if idx < 0:
            idx = 0
        if idx >= n_times:
            idx = n_times - 1
        ax_profile.plot(x_um, c[idx], color="C2")
        if t.size:
            ax_profile.set_title(f"profile at t = {t[idx]:g} s")

    ax_profile.set_xlabel("x (µm)")
    ax_profile.set_ylabel("concentration (m⁻³)")
    ax_profile.grid(True, alpha=0.3)

    plot_concentration_3d(result, ax=ax_3d, cmap=cmap)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return ax_fluxes, ax_profile, ax_3d
