"""
Permeation: 1D hydrogen diffusion through a membrane with recombination boundary conditions.

Solver: Backward Euler (implicit) with nonlinear BCs handled by iterative correction.
Inverse fit: recover G(t) step values from measured downstream pressure (see inverse_fit).
"""

from permeation.diffusion import BE, Parameters
from permeation.materials import (
    constant_G,
    zeros_G,
    step_G,
    multi_step_G,
    refine_steps,
    steps_from_starts,
)
from permeation.plotting import (
    plot_concentration_3d,
    plot_fluxes,
    plot_profiles,
    plot_summary,
)
from permeation.utils import chi_square
from permeation.inverse_fit import (
    simulate_from_step_vals,
    interp_to_meas_grid,
    fit_G_steps,
    fit_G_steps_zoom,
)

__all__ = [
    "BE",
    "Parameters",
    "constant_G",
    "zeros_G",
    "step_G",
    "multi_step_G",
    "refine_steps",
    "steps_from_starts",
    "chi_square",
    "plot_summary",
    "plot_profiles",
    "plot_fluxes",
    "plot_concentration_3d",
    "simulate_from_step_vals",
    "interp_to_meas_grid",
    "fit_G_steps",
    "fit_G_steps_zoom",
]
