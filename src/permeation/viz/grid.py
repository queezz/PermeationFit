"""
Finite-difference grid and stencil visualization for intuition and documentation.
No solver logic or permeation physics. Matplotlib only.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Time-mode semantic labels: (di, dj) -> mathtext (fallback to raw (di,dj))
_TIME_STENCIL_LABELS: dict[tuple[int, int], str] = {
    (0, 0): r"$u^{n+1}$",
    (-1, 0): r"$u^{n}$",
}


def plot_grid_stencil(
    grid_shape: tuple[int, int] = (7, 7),
    center: tuple[int, int] | None = None,
    stencil: list[tuple[int, int]] | None = None,
    *,
    labels: bool = False,
    ax: Any = None,
    title: str | None = None,
    mode: str = "generic",
) -> Any:
    """
    Visualize a regular Cartesian grid with a highlighted finite-difference stencil.
    Grid nodes: light gray hollow; center: solid blue; stencil: solid red. Integer
    indices (i, j). Equal aspect, no ticks/spines. Returns Axes.

    With mode="time", axis annotations (time/space), semantic labels (e.g. u^{n+1}),
    and aspect 1.2 for time-stencil intuition.

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

    # Stencil nodes: green; time mode = slightly smaller (known past)
    ms_stencil = 8 if is_time else 10
    for di, dj in stencil:
        si, sj = ci + di, cj + dj
        if 0 <= si < ni and 0 <= sj < nj:
            ax.plot(
                sj, si, "o", color="#5ca832", fillstyle="full", ms=ms_stencil, zorder=3
            )
            if labels:
                lbl = (
                    _TIME_STENCIL_LABELS.get((di, dj), f"({di},{dj})")
                    if is_time
                    else f"({di},{dj})"
                )
                fs = 26 if is_time else 12
                ax.annotate(
                    lbl,
                    (sj, si),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=fs,
                )

    # Center node: reddish; time mode = slightly larger (unknown n+1)
    ms_center = 12 if is_time else 10
    ax.plot(cj, ci, "o", color="#c73c4c", fillstyle="full", ms=ms_center, zorder=4)

    ax.set_aspect(1.2 if is_time else "equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if is_time:
        ax.annotate(
            "time →",
            (0, 1),
            xycoords="axes fraction",
            xytext=(2, -2),
            textcoords="offset points",
            fontsize=12,
            va="top",
            rotation=90,
        )
        ax.annotate(
            "space →",
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
