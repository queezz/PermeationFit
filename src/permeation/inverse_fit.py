"""
Inverse fit: recover step values of G(t) from measured downstream pressure (pdp).

Given measurement times, measured pdp, step start times, and base Parameters,
fits the step values by least-squares with optional L2 and total-variation
regularization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from permeation.diffusion import BE, Parameters
from permeation.materials import multi_step_G, refine_steps, steps_from_starts


def _bin_scores_from_residual(t_meas, r, edges_frac, T=None, weight=None):
    if T is None:
        t = np.asarray(t_meas)
        edges = np.asarray(edges_frac)
    else:
        t = np.asarray(t_meas)
        edges = np.asarray(edges_frac) * float(T)

    r = np.asarray(r, float)
    if weight is None:
        w = np.ones_like(r)
    else:
        w = np.asarray(weight, float)

    K = len(edges) - 1
    scores = np.zeros(K, float)
    idx = np.searchsorted(edges, t, side="right") - 1
    valid = (idx >= 0) & (idx < K)
    idx = idx[valid]
    rv = r[valid]
    wv = w[valid]
    np.add.at(scores, idx, (rv * rv) * wv)
    return scores


def refine_bins_adaptive(
    edges_frac,
    x_hat,
    t_meas,
    pdp_meas,
    pdp_hat_meas,
    *,
    T=None,
    weight=None,
    max_splits=2,
    min_width_frac=0.02,
    split_strategy="mid",
):
    edges_frac = np.asarray(edges_frac, float)
    x_hat = np.asarray(x_hat, float)
    r = np.asarray(pdp_meas, float) - np.asarray(pdp_hat_meas, float)
    scores = _bin_scores_from_residual(t_meas, r, edges_frac, T=T, weight=weight)
    widths = np.diff(edges_frac)
    eligible = widths >= (2 * min_width_frac)
    cand = np.where(eligible)[0]
    if cand.size == 0:
        return edges_frac, x_hat, scores

    worst = cand[np.argsort(scores[cand])[::-1]]
    worst = worst[:max_splits]
    new_edges = [edges_frac[0]]
    new_x = []

    for k in range(len(x_hat)):
        a = edges_frac[k]
        b = edges_frac[k + 1]
        val = x_hat[k]
        if k in set(worst):
            m = 0.5 * (a + b)
            new_edges.append(m)
            new_edges.append(b)
            new_x.append(val)
            new_x.append(val)
        else:
            new_edges.append(b)
            new_x.append(val)

    new_edges = np.asarray(new_edges, float)
    keep = np.hstack(([True], np.diff(new_edges) > 0))
    new_edges = new_edges[keep]
    new_x = np.asarray(new_x, float)
    return new_edges, new_x, scores


def simulate_from_step_vals(
    step_vals,
    tstart,
    base_params,
    *,
    enforce_zero_after: float | None = None,
):
    """
    Run a permeation simulation with G(t) given by step values at tstart.

    Parameters
    ----------
    step_vals : array-like
        G step values (one per interval starting at each tstart).
    tstart : array-like
        Fractional start times in [0, 1] for each step.
    base_params : Parameters
        Base parameters (ks, kd, D, etc.); G is overridden by the step profile.

    Returns
    -------
    time : ndarray
        Model time grid.
    pdp : ndarray
        Downstream pressure (same length as time).
    G : ndarray
        Incident flux profile (same length as time).
    """
    base_kwargs = base_params.to_dict()
    base_kwargs.pop("G", None)
    base_kwargs.pop("G_generator", None)

    step_vals = np.asarray(step_vals, float)
    tstart = np.asarray(tstart, float)

    if enforce_zero_after is not None:
        tstart = np.append(tstart, enforce_zero_after)
        step_vals = np.append(step_vals, 0.0)

    G_gen = multi_step_G(steps_from_starts(step_vals, tstart))
    params = Parameters(**base_kwargs, G_generator=G_gen)
    result = BE(params)

    return result["time"], result["pdp"], result["G"]


def interp_to_meas_grid(
    t_model: np.ndarray,
    y_model: np.ndarray,
    t_meas: np.ndarray,
) -> np.ndarray:
    """Interpolate model output (t_model, y_model) onto measurement times t_meas."""
    return np.interp(t_meas, t_model, y_model)


def fit_G_steps(
    t_meas: np.ndarray | list[float],
    pdp_meas: np.ndarray | list[float],
    tstart: np.ndarray | list[float],
    base_params: Parameters,
    x0: np.ndarray | list[float],
    *,
    bounds: tuple[float, float] = (0.0, np.inf),
    weight: np.ndarray | None = None,
    reg_l2: float = 0.0,
    reg_tv: float = 0.0,
    verbose: int = 2,
    G_zero_after: float | None = None,
    **least_squares_kwargs: Any,
) -> dict[str, Any]:
    """
    Least-squares fit of a piecewise-constant incident flux G(t).

    Fits step amplitudes x for fixed step start times tstart by matching
    the downstream pressure pdp(t). Optional L2 and total-variation
    regularization can be applied to the step values.

    Parameters
    ----------
    t_meas : array-like
        Measurement time grid.
    pdp_meas : array-like
        Measured downstream pressure.
    tstart : array-like
        Step start times (fractional); defines the piecewise structure.
    base_params : Parameters
        Physical parameters; G(t) is determined by the fit.
    x0 : array-like
        Initial guess for step values.
    bounds : (lb, ub)
        Bounds for step values.
    weight : array-like, optional
        Per-point weights for residuals.
    reg_l2 : float, optional
        L2 regularization strength on x.
    reg_tv : float, optional
        Total-variation regularization on diff(x).
    verbose : int, optional
        Verbosity level for least_squares.
    **least_squares_kwargs
        Passed to scipy.optimize.least_squares.

    Returns
    -------
    dict
        x_hat, result, t_model, pdp_hat, G_hat, tstart
    """

    t_meas = np.asarray(t_meas, float)
    pdp_meas = np.asarray(pdp_meas, float)

    if weight is None:
        w = 1.0
    else:
        w = np.asarray(weight, float)
        if w.shape != pdp_meas.shape:
            raise ValueError("weight must have same shape as pdp_meas")

    tstart = np.asarray(tstart, float)

    def residuals(x):
        t_model, pdp_model, _ = simulate_from_step_vals(
            x,
            tstart,
            base_params,
            enforce_zero_after=G_zero_after,
        )
        pdp_model_i = interp_to_meas_grid(t_model, pdp_model, t_meas)

        r = (pdp_model_i - pdp_meas) * w

        if reg_l2 > 0:
            r = np.concatenate([r, np.sqrt(reg_l2) * x])

        if reg_tv > 0:
            dx = np.diff(x)
            r = np.concatenate([r, np.sqrt(reg_tv) * dx])

        return r

    res = least_squares(
        residuals,
        x0=np.asarray(x0, float),
        bounds=bounds,
        verbose=verbose,
        **least_squares_kwargs,
    )

    x_hat = res.x
    t_model, pdp_hat, G_hat = simulate_from_step_vals(
        x_hat, tstart, base_params, enforce_zero_after=G_zero_after
    )

    return {
        "x_hat": x_hat,
        "result": res,
        "t_model": t_model,
        "pdp_hat": pdp_hat,
        "G_hat": G_hat,
        "tstart": tstart,
    }


def fit_G_steps_zoom(
    t_meas: np.ndarray | list[float],
    pdp_meas: np.ndarray | list[float],
    base_params: Parameters,
    initial_guess: float,
    n_levels: int,
    *,
    bounds: tuple[float, float] = (0.0, np.inf),
    weight: np.ndarray | None = None,
    reg_l2: float = 1e-6,
    reg_tv: float = 1e-3,
    max_nfev: int = 200,
    verbose: int = 1,
    **fit_G_steps_kwargs: Any,
) -> dict[str, Any]:
    """
    Multi-level "zoom" fit: start with one step, refine by splitting steps each level.

    Level 0: one step over [0, 1]. Each level doubles the number of steps via
    refine_steps (split each step into two). Fitted values from the previous
    level are used as initial guess for the next.

    Parameters
    ----------
    t_meas : array-like
        Measurement time grid.
    pdp_meas : array-like
        Measured downstream pressure at t_meas.
    base_params : Parameters
        Base physical parameters.
    initial_guess : float
        Initial step value for the single step at level 0.
    n_levels : int
        Number of refinement levels (0 = one step only).
    bounds : tuple of (lb, ub)
        Bounds for each step value.
    weight : array-like, optional
        Per-point weights; same shape as pdp_meas.
    reg_l2, reg_tv : float
        L2 and total-variation regularization (used at every level).
    max_nfev : int
        Max function evaluations per level (passed to least_squares).
    verbose : int
        Verbosity for least_squares (0, 1, 2).
    **fit_G_steps_kwargs
        Passed through to fit_G_steps each level (e.g. ftol, xtol).

    Returns
    -------
    dict with keys
        history : list of dict
            One fit result dict per level (same structure as fit_G_steps return).
        tstart : ndarray
            Step start times from the last level.
        x_hat : ndarray
            Fitted step values from the last level.
    """
    tstart = np.array([0.0], dtype=float)
    x0 = np.array([float(initial_guess)])

    history: list[dict[str, Any]] = []

    for level in range(n_levels):
        out = fit_G_steps(
            t_meas,
            pdp_meas,
            tstart,
            base_params,
            x0,
            bounds=bounds,
            weight=weight,
            reg_l2=reg_l2,
            reg_tv=reg_tv,
            verbose=verbose,
            max_nfev=max_nfev,
            **fit_G_steps_kwargs,
        )
        history.append(out)

        # prepare next level: double steps via refine_steps
        tstart, x0 = refine_steps(tstart, out["x_hat"])

    return {
        "history": history,
        "tstart": history[-1]["tstart"],
        "x_hat": history[-1]["x_hat"],
    }


def fit_with_adaptive_bins(
    t_meas: np.ndarray | list[float],
    pdp_meas: np.ndarray | list[float],
    edges_frac: np.ndarray | list[float],
    x0: np.ndarray | list[float],
    base_params: Parameters,
    *,
    T: float | None = None,
    weight: np.ndarray | None = None,
    max_bins: int = 64,
    max_refinement_depth: int = 20,
    max_splits: int = 2,
    min_width_frac: float = 0.02,
    split_strategy: str = "mid",
    bounds: tuple[float, float] = (0.0, np.inf),
    reg_l2: float = 0.0,
    reg_tv: float = 0.0,
    G_zero_after: float | None = None,
    verbose: int = 2,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    edges_frac = np.asarray(edges_frac, float)
    x0 = np.asarray(x0, float)
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    out = None
    x_hat = x0

    for _ in range(max(1, max_refinement_depth)):
        tstart = edges_frac[:-1].copy()
        out = fit_G_steps(
            t_meas,
            pdp_meas,
            tstart,
            base_params,
            x0,
            bounds=bounds,
            weight=weight,
            reg_l2=reg_l2,
            reg_tv=reg_tv,
            verbose=verbose,
            G_zero_after=G_zero_after,
            **fit_kwargs,
        )
        x_hat = out["x_hat"]
        pdp_hat_meas = interp_to_meas_grid(out["t_model"], out["pdp_hat"], t_meas)
        new_edges, new_x, scores = refine_bins_adaptive(
            edges_frac,
            x_hat,
            t_meas,
            pdp_meas,
            pdp_hat_meas,
            T=T,
            weight=weight,
            max_splits=max_splits,
            min_width_frac=min_width_frac,
            split_strategy=split_strategy,
        )
        history.append((edges_frac.copy(), x_hat.copy(), scores))

        no_split = len(new_x) == len(x_hat)
        at_max_bins = len(new_x) >= max_bins
        if no_split or at_max_bins:
            break
        edges_frac = new_edges
        x0 = new_x

    return {
        "edges_frac": edges_frac,
        "x_hat": x_hat,
        "history": history,
        "result": out,
        "tstart": out["tstart"],
    }
