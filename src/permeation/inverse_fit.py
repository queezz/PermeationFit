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


def simulate_from_step_vals(
    step_vals: np.ndarray | list[float],
    tstart: np.ndarray | list[float],
    base_params: Parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    **least_squares_kwargs: Any,
) -> dict[str, Any]:
    """
    Fit step values of G(t) to measured downstream pressure by least squares.

    The unknown vector x is the step values; step times are fixed by tstart.
    Optionally penalize L2 norm of x and/or total variation (smoothness) of x.

    Parameters
    ----------
    t_meas : array-like
        Measurement time grid (fractional or absolute; must match model time axis).
    pdp_meas : array-like
        Measured downstream pressure at t_meas.
    tstart : array-like
        Fractional start times for each step (length = number of steps).
    base_params : Parameters
        Base physical parameters; G is determined by the fit.
    x0 : array-like
        Initial guess for step values (length = len(tstart)).
    bounds : tuple of (lb, ub)
        Lower and upper bounds for each step value. Default (0, inf).
    weight : array-like, optional
        Per-point weights; same shape as pdp_meas. Default unweighted.
    reg_l2 : float
        L2 regularization strength on x. Default 0.
    reg_tv : float
        Total-variation regularization strength on diff(x). Default 0.
    verbose : int
        Verbosity for scipy.optimize.least_squares (0, 1, 2).
    **least_squares_kwargs
        Passed through to least_squares (e.g. ftol, xtol, max_nfev).

    Returns
    -------
    dict with keys
        x_hat : ndarray
            Fitted step values.
        result : scipy.optimize.OptimizeResult
            Full least_squares result.
        t_model : ndarray
            Model time grid.
        pdp_hat : ndarray
            Fitted pdp on model grid.
        G_hat : ndarray
            Recovered G(t) on model grid.
        tstart : ndarray
            Step start times (copy of input).
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

    def residuals(x: np.ndarray) -> np.ndarray:
        _, pdp_model, _ = simulate_from_step_vals(x, tstart, base_params)
        pdp_model_i = interp_to_meas_grid(_, pdp_model, t_meas)

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
    t_model, pdp_hat, G_hat = simulate_from_step_vals(x_hat, tstart, base_params)

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
