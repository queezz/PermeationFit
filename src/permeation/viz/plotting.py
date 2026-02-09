"""
Plotting helpers for permeation solver output and inverse-fit workflow.

Operate only on result dictionaries. No solver internals or global state.
Requires matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# MARK: profiles
def plot_profiles(
    result: dict[str, Any],
    time_idx: int | Sequence[int] | None = None,
    ax: Any = None,
    savepath: str | None = None,
    show_colorbar: bool = True,
    cbar_kwargs: dict[str, Any] | None = None,
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
    show_colorbar : bool, optional
        If True, add a colorbar for time (default True).
    cbar_kwargs : dict, optional
        Extra keyword args passed to plt.colorbar when show_colorbar is True.

    Returns
    -------
    matplotlib axes
    """
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
    if show_colorbar:
        cbar = plt.colorbar(sm, ax=ax, **(cbar_kwargs or {}))
        cbar.set_label("time (s)")
    ax.grid(True, alpha=0.3)
    ax.get_figure().subplots_adjust(left=0.22)
    if savepath:
        ax.get_figure().savefig(savepath, bbox_inches="tight")
    return ax


# MARK: fluxes
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
    fluxes = result["fluxes"]
    if ax is None:
        _, ax = plt.subplots()

    t = fluxes["time"]
    ax.plot(t, fluxes["rel"], label="inlet (desorbed)", color="C0", linewidth=2)
    ax.plot(t, fluxes["perm"], label="outlet (permeation)", color="C1", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("flux (m⁻² s⁻¹)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.get_figure().subplots_adjust(left=0.22)
    if savepath:
        ax.get_figure().savefig(savepath, bbox_inches="tight")
    return ax


# MARK: 3D
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
    from matplotlib import cm as mpl_cm

    x = np.asarray(result["x"])
    t = np.asarray(result["time"])
    c = np.asarray(result["c"], dtype=float)
    x_um = x / 1e-6
    X, Y = np.meshgrid(x_um, t)

    c_max = float(np.max(c))
    pwr = int(np.round(np.log10(c_max))) if c_max > 0 else 0
    scale = 10.0**pwr
    a = c / scale if scale > 0 else c

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    cmap_obj = mpl_cm.get_cmap(cmap)
    surf = ax.plot_surface(
        X,
        Y,
        a,
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


# MARK: solver summary
def plot_summary(
    result,
    profile_time_idx=None,
    cmap="inferno",
    figsize=(15, 8),
    savepath=None,
):
    """
    Multi-panel summary: fluxes, spatial profiles, 3D concentration surface.
    For BE() solver result.
    """
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        width_ratios=(1.0, 1.6),
        height_ratios=(1.0, 1.0),
        wspace=0.35,
        hspace=0.2,
    )

    ax_flux = fig.add_subplot(gs[0, 0])
    ax_prof = fig.add_subplot(gs[1, 0])

    # span both rows
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")

    # --- Fluxes ---
    fluxes = result["fluxes"]
    flux_max = float(np.nanmax(np.abs([fluxes["rel"].max(), fluxes["perm"].max()])))
    flux_pwr = int(np.floor(np.log10(flux_max))) if flux_max > 0 else 0
    flux_scale = 10.0**flux_pwr
    fluxes_scaled = fluxes.copy()
    if flux_scale > 0:
        fluxes_scaled["rel"] = fluxes_scaled["rel"] / flux_scale
        fluxes_scaled["perm"] = fluxes_scaled["perm"] / flux_scale
    plot_fluxes({**result, "fluxes": fluxes_scaled}, ax=ax_flux)
    ax_flux.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        frameon=False,
        borderaxespad=0.0,
    )
    ax_flux.set_ylabel(f"flux × 10^{flux_pwr} (m⁻² s⁻¹)")

    x = np.asarray(result["x"]) / 1e-6
    t = np.asarray(result["time"])
    c = np.asarray(result["c"])

    # --- Incident flux G (right axis on panel a) ---
    G = result.get("G")
    if G is None:
        G = result.get("params", {}).get("G")
    if G is None:
        G = np.zeros_like(t)
    G = np.asarray(G, dtype=float)
    g_max = float(np.nanmax(np.abs(G))) if G.size else 0.0
    g_pwr = int(np.floor(np.log10(g_max))) if g_max > 0 else 0
    g_scale = 10.0**g_pwr
    G_scaled = G / g_scale if g_scale > 0 else G
    ax_g = ax_flux.twinx()
    ax_g.plot(t, G_scaled, color="k", linestyle="--", label="incident (G)")
    ax_g.set_ylabel(f"G × 10^{g_pwr} (m⁻² s⁻¹)")
    ax_g.grid(False)

    # --- Profiles (2D) ---
    prof_max = float(np.nanmax(np.abs(c))) if c.size else 0.0
    prof_pwr = int(np.floor(np.log10(prof_max))) if prof_max > 0 else 0
    prof_scale = 10.0**prof_pwr
    c_scaled = c / prof_scale if prof_scale > 0 else c
    plot_profiles(
        {**result, "c": c_scaled},
        time_idx=profile_time_idx,
        ax=ax_prof,
        show_colorbar=True,
        cbar_kwargs={"shrink": 0.75},
    )
    ax_prof.set_ylabel(f"concentration × 10^{prof_pwr} (m⁻³)")

    # --- 3D surface ---
    X, Y = np.meshgrid(x, t)
    cmax = float(np.max(c))
    pwr = int(np.round(np.log10(cmax))) if cmax > 0 else 0
    Z = c / (10**pwr)

    surf = ax_3d.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=False,
    )

    ax_3d.view_init(25, 40)
    ax_3d.set_xlabel("d (µm)", labelpad=12)
    ax_3d.set_ylabel("time (s)", labelpad=12)
    ax_3d.zaxis.set_rotate_label(False)
    ax_3d.set_zlabel(f"c × 10^{pwr} (H/m³)", labelpad=10, rotation=90)

    # --- panel labels ---
    ax_flux.text(
        -0.1,
        1.1,
        "a.",
        transform=ax_flux.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax_prof.text(
        -0.1,
        1.1,
        "b.",
        transform=ax_prof.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax_3d.text2D(
        0.02,
        0.98,
        "c.",
        transform=ax_3d.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")

    fig.subplots_adjust(top=0.92)
    return ax_flux, ax_3d, ax_prof


# MARK: inverse-fit plotting
def plot_G(
    t_meas: np.ndarray,
    pdp_meas: np.ndarray,
    *,
    t_true: np.ndarray | None = None,
    pdp_true: np.ndarray | None = None,
    G_true: np.ndarray | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """
    Two-panel: pdp and G (true/input). Use before fitting to inspect synthetic data.
    For real data, only pdp_meas is shown.
    """
    has_true = t_true is not None and pdp_true is not None
    has_G = t_true is not None and G_true is not None

    if has_G:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    else:
        fig, ax1 = plt.subplots()
        ax2 = None

    ax1.plot(t_meas, pdp_meas, "o", color="#8f6ed4", ms=2, label="meas")
    if has_true:
        ax1.plot(t_true, pdp_true, "-", color="#8f6ed4", label="true")
    ax1.set_ylabel("pdp")
    ax1.legend(frameon=False)

    if has_G and ax2 is not None:
        ax2.step(t_true, G_true, where="post", lw=2, color="#8f6ed4", label="true G")
        ax2.set_ylabel("G")
        ax2.set_xlabel("time")
        ax2.legend(frameon=False)
    else:
        ax1.set_xlabel("time")

    plt.tight_layout()
    return fig


# MARK: INV Sum
def plot_inverse_summary(
    zoom: dict[str, Any],
    t_meas: np.ndarray,
    pdp_meas: np.ndarray,
    *,
    t_true: np.ndarray | None = None,
    pdp_true: np.ndarray | None = None,
    G_true: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Two-panel: measured vs fitted pdp and recovered G across zoom levels.
    Optional t_true, pdp_true, G_true for synthetic-data comparison.
    """
    hist = zoom["history"]
    fit = hist[-1]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if t_true is not None and pdp_true is not None:
        ax1.plot(t_true, pdp_true, "-", color="#8f6ed4", label="true")
    ax1.plot(t_meas, pdp_meas, "o", color="#8f6ed4", ms=2, label="meas")

    n_hist = len(hist)
    cmap = plt.get_cmap("autumn")
    norm = mcolors.Normalize(vmin=0, vmax=max(1, n_hist - 1))

    for k, h in enumerate(hist):
        color = cmap(norm(k))
        ax1.plot(h["t_model"], h["pdp_hat"], "-", lw=1.2, color=color)
        ax2.step(h["t_model"], h["G_hat"], where="post", lw=1.2, color=color)

    ax1.plot(
        fit["t_model"],
        fit["pdp_hat"],
        "-",
        lw=2.5,
        color="#454231",
        label="fit (final)",
    )
    ax2.step(
        fit["t_model"],
        fit["G_hat"],
        where="post",
        lw=2.5,
        color="#454231",
        label="G (final)",
    )
    if t_true is not None and G_true is not None:
        ax2.step(t_true, G_true, where="post", lw=2, color="#8f6ed4", label="true G")

    ax1.set_ylabel("pdp")
    ax1.legend(frameon=False)
    ax2.set_ylabel("G")
    ax2.set_xlabel("time")
    ax2.legend(frameon=False)
    plt.tight_layout()
    return fig, (ax1, ax2)


# MARK: zoom frm
def plot_zoom_frame(
    zoom: dict[str, Any],
    t_meas: np.ndarray,
    pdp_meas: np.ndarray,
    level: int,
    *,
    t_true: np.ndarray | None = None,
    pdp_true: np.ndarray | None = None,
    G_true: np.ndarray | None = None,
    active_color: str = "#b24a3a",
    history_color: str = "0.7",
    **kwargs: Any,
) -> plt.Figure:
    """
    Single-level frame: previous levels in gray, active level highlighted.
    For animation/sequence visualization.
    """
    hist = zoom["history"]
    if level < 0 or level >= len(hist):
        raise ValueError(f"level must be in [0, {len(hist)-1}]")
    h = hist[level]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(t_meas, pdp_meas, "o", ms=2, color="#8f6ed4")
    if t_true is not None and pdp_true is not None:
        ax1.plot(t_true, pdp_true, "-", color="#8f6ed4")
    if t_true is not None and G_true is not None:
        ax2.step(t_true, G_true, where="post", lw=1.5, color="#8f6ed4", label="true G")

    for j in range(level):
        h_prev = hist[j]
        ax1.plot(h_prev["t_model"], h_prev["pdp_hat"], lw=0.8, color=history_color)
        ax2.step(
            h_prev["t_model"],
            h_prev["G_hat"],
            where="post",
            lw=0.8,
            color=history_color,
        )

    ax1.plot(h["t_model"], h["pdp_hat"], lw=2.0, color=active_color)
    ax2.step(h["t_model"], h["G_hat"], where="post", lw=2.0, color=active_color)

    t_all = np.concatenate([x["t_model"] for x in hist])
    pdp_all = np.concatenate([x["pdp_hat"] for x in hist] + [pdp_meas])
    G_all = np.concatenate([x["G_hat"] for x in hist])
    if t_true is not None and G_true is not None:
        G_all = np.concatenate([G_all, G_true])

    tlim = (t_all.min(), t_all.max())
    pdp_pad = 0.05 * (pdp_all.max() - pdp_all.min()) if pdp_all.size else 0
    G_pad = 0.05 * (G_all.max() - G_all.min()) if G_all.size else 0
    ax1.set_xlim(tlim)
    ax1.set_ylim(pdp_all.min() - pdp_pad, pdp_all.max() + pdp_pad)
    ax2.set_ylim(G_all.min() - G_pad, G_all.max() + G_pad)

    ax1.set_ylabel("pdp")
    ax2.set_ylabel("G")
    ax2.set_xlabel("time")
    plt.tight_layout()
    return fig


# MARK: Conve Hi
def plot_convergence_history(
    zoom: dict[str, Any],
    **kwargs: Any,
) -> plt.Figure:
    """Cost vs zoom level."""
    hist = zoom["history"]
    costs = [h["result"].cost for h in hist]
    levels = list(range(len(hist)))
    fig, ax = plt.subplots()
    ax.semilogy(levels, costs, "o-", color="#454231")
    ax.set_xlabel("zoom level")
    ax.set_ylabel("cost")
    ax.set_title("Convergence")
    plt.tight_layout()
    return fig


# MARK: Export St
def export_zoom_states_frames(
    zoom: dict[str, Any],
    t_meas: np.ndarray,
    pdp_meas: np.ndarray,
    base_params: Any,
    *,
    outdir: str = "frames",
    fade: float = 0.25,
    max_hist: int = 20,
    active_color: str = "#b24a3a",
    history_color: str = "0.7",
    dpi: int = 150,
    figsize: Sequence[float] = (6, 4),
    show_title: bool = True,
    enforce_zero_after: float | None = None,
) -> list[str]:
    """
    Save one PNG frame per optimizer state. Groups by zoom level.
    Requires zoom["states"] (run fit with save_states=True).
    Returns list of saved file paths. Deterministic, batch-safe.
    """
    from permeation.inverse.inverse_fit import simulate_from_step_vals

    states = zoom.get("states")
    if states is None:
        raise ValueError("zoom['states'] is None; run fit with save_states=True")

    t_meas = np.asarray(t_meas, float)
    pdp_meas = np.asarray(pdp_meas, float)
    saved: list[str] = []

    # Group states by level (states are sequential: level 0, then 1, ...)
    by_level: dict[int, list[dict]] = {}
    for s in states:
        lev = int(s["level"])
        by_level.setdefault(lev, []).append(s)

    for level, lev_states in sorted(by_level.items()):
        level_dir = Path(outdir) / f"level_{level}"
        level_dir.mkdir(parents=True, exist_ok=True)

        # Precompute model curves for all states at this level
        curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for s in lev_states:
            t_model, pdp_hat, G_hat = simulate_from_step_vals(
                s["x"],
                s["tstart"],
                base_params,
                enforce_zero_after=enforce_zero_after,
            )
            curves.append((t_model, pdp_hat, G_hat))

        t_all = np.concatenate([c[0] for c in curves] + [t_meas])
        pdp_all = np.concatenate([c[1] for c in curves] + [pdp_meas])
        G_all = np.concatenate([c[2] for c in curves])
        tlim = (t_all.min(), t_all.max())
        pdp_pad = 0.05 * (pdp_all.max() - pdp_all.min()) if pdp_all.size else 0
        G_pad = 0.05 * (G_all.max() - G_all.min()) if G_all.size else 0

        for k in range(len(lev_states)):
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)

            ax1.plot(t_meas, pdp_meas, "o", color="black", ms=2)

            j0 = max(0, k - max_hist)

            for j in range(j0, k):
                # for j in range(k):
                age = k - j
                alpha = 0.25 + 0.75 * np.exp(-fade * age)
                # alpha = float(fade ** (k - j))
                t_m, p_m, g_m = curves[j]
                ax1.plot(t_m, p_m, "-", lw=0.8, color=history_color, alpha=alpha)
                ax2.step(
                    t_m, g_m, where="post", lw=0.8, color=history_color, alpha=alpha
                )

            t_m, p_m, g_m = curves[k]
            ax1.plot(t_m, p_m, "-", lw=2.0, color=active_color)
            ax2.step(t_m, g_m, where="post", lw=2.0, color=active_color)

            ax1.set_xlim(tlim)
            ax1.set_ylim(pdp_all.min() - pdp_pad, pdp_all.max() + pdp_pad)
            ax2.set_ylim(G_all.min() - G_pad, G_all.max() + G_pad)
            ax1.set_ylabel("pdp")
            ax2.set_ylabel("G")
            ax2.set_xlabel("time")

            if show_title:
                cost = lev_states[k].get("cost", float("nan"))
                fig.suptitle(f"Level {level}, iter {k}, cost={cost:.3e}")

            plt.tight_layout()
            path = level_dir / f"frame_{k:03d}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved.append(str(path))

    return saved
