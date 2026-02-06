"""
Permeation: 1D hydrogen diffusion through a membrane with recombination boundary conditions.

Solver: Backward Euler (implicit) with nonlinear BCs handled by iterative correction.
"""

from permeation.diffusion import BE, parameters
from permeation.plotting import plot_fluxes, plot_profiles
from permeation.utils import chi_square

__all__ = ["BE", "parameters", "chi_square", "plot_profiles", "plot_fluxes"]
