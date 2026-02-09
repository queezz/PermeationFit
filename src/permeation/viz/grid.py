"""
Finite-difference grid and stencil visualization for intuition and documentation.
No solver logic or permeation physics. Matplotlib only.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COLORS_TIME = {
    "known": "#5ca832",   # n (past, given)
    "unknown": "#c7a63c",  # n+1 (future, coupled)
    "focus": "#c73c4c",    # u_j^{n+1} (center unknown)
}


def _time_role(dt: int, dj: int) -> str:
    """Time-role for coloring: known (n), unknown (n+1), focus (center n+1)."""
    if dt == -1:
        return "known"
    if dt == 0 and dj == 0:
        return "focus"
    if dt == 0:
        return "unknown"
    return "unknown"


def _stencil_label_fd(dt: int, dj: int) -> str:
    """Convert (dt, dj) offset to FD semantic LaTeX: u^{time}_{space}. dt=time, dj=space."""
    if dt == 0:
        sup = "n+1"
    elif dt == -1:
        sup = "n"
    else:
        sup = f"n{1 + dt:+d}"
    if dj == 0:
        sub = "j"
    else:
        sub = f"j{dj:+d}"
    return rf"$u^{{{sup}}}_{{{sub}}}$"


def plot_grid_stencil(
    grid_shape: tuple[int, int] = (7, 7),
    center: tuple[int, int] | None = None,
    stencil: list[tuple[int, int]] | None = None,
    *,
    labels: bool | str = False,
    ax: Any = None,
    title: str | None = None,
    mode: str = "generic",
) -> Any:
    """
    Visualize a regular Cartesian grid with a highlighted finite-difference stencil.
    Grid nodes: light gray hollow; center: solid blue; stencil: solid red. Integer
    indices (i, j). Equal aspect, no ticks/spines. Returns Axes.

    With mode="time", axis annotations (time/space), semantic FD labels (e.g. u^{n+1}_j),
    and aspect 1.2 for time-stencil intuition. labels=True shows semantic labels;
    labels="offset" shows raw (dt, dx) for debugging.

    Example
    -------
    >>> plot_grid_stencil(
    ...     grid_shape=(7,7),
    ...     stencil=[(0,0),(1,0),(-1,0),(0,1),(0,-1)],
    ...     labels=True,
    ... )
    """
    ni, nj = grid_shape
    ci, cj = center if center is not None else (ni // 2, nj // 2)
    stencil = stencil if stencil is not None else [(0, 0)]
    is_time = mode == "time"

    if ax is None:
        _, ax = plt.subplots()

    # Grid: (i, j) -> x = j, y = i. Time mode: smaller hollow nodes
    ms_grid = 10 if is_time else 12
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    ax.plot(
        jj.ravel(),
        ii.ravel(),
        "o",
        color="lightgray",
        fillstyle="none",
        ms=ms_grid,
        markeredgewidth=3,
    )

    # Stencil nodes: time mode = color by role (known/unknown/focus); generic = single color
    ms_stencil = 8 if is_time else 10
    for di, dj in stencil:
        si, sj = ci + di, cj + dj
        if 0 <= si < ni and 0 <= sj < nj:
            color = COLORS_TIME[_time_role(di, dj)] if is_time else "#5ca832"
            ax.plot(
                sj, si, "o", color=color, fillstyle="full", ms=ms_stencil, zorder=3
            )
            if labels:
                if labels == "offset":
                    lbl = f"({di},{dj})"
                elif is_time:
                    lbl = _stencil_label_fd(di, dj)
                else:
                    lbl = f"({di},{dj})"
                fs = 26 if is_time else 12
                ax.annotate(
                    lbl,
                    (sj, si),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=fs,
                )

    # Center node: time mode = focus color from role; generic = same as before
    ms_center = 12 if is_time else 10
    center_color = COLORS_TIME[_time_role(0, 0)] if is_time else "#c73c4c"
    ax.plot(cj, ci, "o", color=center_color, fillstyle="full", ms=ms_center, zorder=4)

    ax.set_aspect(1.2 if is_time else "equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if is_time:
        ax.annotate(
            "time index n →",
            (-0.03, 1),
            xycoords="axes fraction",
            xytext=(2, -2),
            textcoords="offset points",
            fontsize=12,
            va="top",
            rotation=90,
        )
        ax.annotate(
            "space index j →",
            (1, -0.02),
            xycoords="axes fraction",
            xytext=(-2, 0),
            textcoords="offset points",
            fontsize=12,
            ha="right",
            va="top",
        )
    if title is not None:
        ax.set_title(title)
    elif is_time:
        ax.set_title("Backward Euler (implicit time stencil)")
    return ax
