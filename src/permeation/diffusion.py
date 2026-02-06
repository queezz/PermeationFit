"""
High-level diffusion API: Backward Euler permeation solver and default parameters.
"""

from __future__ import annotations

from permeation.materials import parameters
from permeation.backward_euler import BE

__all__ = ["BE", "parameters"]
