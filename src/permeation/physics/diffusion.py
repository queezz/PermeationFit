"""
High-level diffusion API: Backward Euler permeation solver and Parameters.
"""

from __future__ import annotations

from permeation.physics.backward_euler import BE
from permeation.physics.materials import Parameters

__all__ = ["BE", "Parameters"]
