"""Workflow for inverse-fitting G(t) step values from measured pdp."""

from __future__ import annotations

from typing import Any

import numpy as np

from permeation.physics.materials import Parameters
from permeation.inverse.inverse_fit import fit_G_steps_zoom, simulate_from_step_vals


class InverseFitWorkflow:
    """
    Holds measurement data and base params, runs zoom fit, delegates plotting.
    No physics or solver logic—wraps permeation.inverse.inverse_fit.
    """

    def __init__(
        self,
        t_meas: np.ndarray,
        pdp_meas: np.ndarray,
        base_params: Parameters,
        *,
        t_true: np.ndarray | None = None,
        pdp_true: np.ndarray | None = None,
        G_true: np.ndarray | None = None,
    ):
        """
        Construct from experimental or real data.

        t_meas, pdp_meas: measurement grid and downstream pressure.
        base_params: physical parameters (ks, kd, D). t_true, pdp_true, G_true optional (for viz).
        """
        self.t_meas = np.asarray(t_meas, float)
        self.pdp_meas = np.asarray(pdp_meas, float)
        self.base_params = base_params
        self.t_true = np.asarray(t_true, float) if t_true is not None else None
        self.pdp_true = np.asarray(pdp_true, float) if pdp_true is not None else None
        self.G_true = np.asarray(G_true, float) if G_true is not None else None
        self._enforce_zero_after: float | None = None
        self._result: dict[str, Any] | None = None

    @classmethod
    def from_synthetic(
        cls,
        tstart: np.ndarray | list[float],
        true_vals: np.ndarray | list[float],
        base_params: Parameters,
        *,
        noise_rel: float = 0.02,
        rng: np.random.Generator | int | None = None,
        enforce_zero_after: float | None = None,
    ) -> InverseFitWorkflow:
        """
        Build workflow from synthetic data: forward-simulate then add noise.

        tstart: step start times (frac); true_vals: step values.
        noise_rel: noise level relative to max pdp. rng: seed or Generator.
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        elif rng is None:
            rng = np.random.default_rng(0)
        t_true, pdp_true, G_true = simulate_from_step_vals(
            np.asarray(true_vals),
            np.asarray(tstart),
            base_params,
            enforce_zero_after=enforce_zero_after,
        )
        noise = noise_rel * np.max(pdp_true) * rng.normal(size=pdp_true.shape)
        pdp_meas = pdp_true + noise
        wf = cls(
            t_meas=t_true.copy(),
            pdp_meas=pdp_meas,
            base_params=base_params,
            t_true=t_true,
            pdp_true=pdp_true,
            G_true=G_true,
        )
        wf._enforce_zero_after = enforce_zero_after
        return wf

    def fit(
        self,
        n_levels: int,
        *,
        initial_guess: float = 0.5,
        bounds: tuple[float, float] = (0.0, 2.0),
        reg_l2: float = 1e-6,
        reg_tv: float = 1e-3,
        max_nfev: int = 200,
        verbose: int = 1,
        save_states: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run fit_G_steps_zoom and store result. Passes kwargs to permeation.
        Returns the zoom result dict.
        """
        kw = dict(kwargs)
        if "G_zero_after" not in kw and self._enforce_zero_after is not None:
            kw["G_zero_after"] = self._enforce_zero_after
        self._result = fit_G_steps_zoom(
            t_meas=self.t_meas,
            pdp_meas=self.pdp_meas,
            base_params=self.base_params,
            initial_guess=initial_guess,
            n_levels=n_levels,
            bounds=bounds,
            reg_l2=reg_l2,
            reg_tv=reg_tv,
            max_nfev=max_nfev,
            verbose=verbose,
            save_states=save_states,
            **kw,
        )
        return self._result

    @property
    def result(self) -> dict[str, Any] | None:
        """Last fit result or None if fit not run."""
        return self._result

    def plot(
        self,
        kind: str = "summary",
        level: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Delegate to permeation.viz.plotting. kind: 'G' | 'summary' | 'zoom_frame' | 'convergence'.
        'G' runs before fitting; others require fit() first. For 'zoom_frame', pass level (int).
        Returns matplotlib figure/axes per viz.
        """
        from permeation.viz.plotting import (
            plot_G,
            plot_convergence_history,
            plot_inverse_summary,
            plot_zoom_frame,
        )

        if kind == "G":
            return plot_G(
                self.t_meas,
                self.pdp_meas,
                t_true=self.t_true,
                pdp_true=self.pdp_true,
                G_true=self.G_true,
                **kwargs,
            )

        if self._result is None:
            raise RuntimeError("Run fit() before plotting")

        zoom = self._result
        if kind == "summary":
            return plot_inverse_summary(
                zoom,
                self.t_meas,
                self.pdp_meas,
                t_true=self.t_true,
                pdp_true=self.pdp_true,
                G_true=self.G_true,
                **kwargs,
            )
        if kind == "zoom_frame":
            if level is None:
                raise ValueError("kind='zoom_frame' requires level")
            return plot_zoom_frame(
                zoom,
                self.t_meas,
                self.pdp_meas,
                level,
                t_true=self.t_true,
                pdp_true=self.pdp_true,
                G_true=self.G_true,
                **kwargs,
            )
        if kind == "convergence":
            return plot_convergence_history(zoom, **kwargs)
        raise ValueError(f"Unknown plot kind: {kind}")

    def export_frames(self, **kwargs: Any) -> list[str]:
        """
        Export zoom-state frames via export_zoom_states_frames. Passes
        enforce_zero_after from workflow when set. Requires save_states=True.
        """
        from permeation.viz.plotting import export_zoom_states_frames

        if self._result is None:
            raise RuntimeError("Run fit() with save_states=True before export_frames")
        kw = dict(kwargs)
        if "enforce_zero_after" not in kw and self._enforce_zero_after is not None:
            kw["enforce_zero_after"] = self._enforce_zero_after
        return export_zoom_states_frames(
            self._result,
            self.t_meas,
            self.pdp_meas,
            self.base_params,
            **kw,
        )
