"""
Permeation: 1D hydrogen diffusion through a membrane with recombination boundary conditions.

Solver: Backward Euler (implicit) with nonlinear BCs handled by iterative correction.
"""

from permeation.diffusion import BE, Parameters
from permeation.materials import (
    constant_G,
    zeros_G,
    step_G,
    multi_step_G,
    steps_from_starts,
)
from permeation.plotting import (
    plot_concentration_3d,
    plot_fluxes,
    plot_profiles,
    plot_summary,
)
from permeation.utils import chi_square

__all__ = [
    "BE",
    "Parameters",
    "constant_G",
    "zeros_G",
    "step_G",
    "multi_step_G",
    "steps_from_starts",
    "chi_square",
    "plot_summary",
    "plot_profiles",
    "plot_fluxes",
    "plot_concentration_3d",
]
