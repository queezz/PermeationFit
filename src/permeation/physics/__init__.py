"""Physics: diffusion solver, materials, parameters."""

from permeation.physics.diffusion import BE, Parameters
from permeation.physics.materials import (
    constant_G,
    zeros_G,
    step_G,
    multi_step_G,
    refine_steps,
    steps_from_starts,
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
]
