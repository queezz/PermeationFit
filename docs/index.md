# Numerical solutions for hydrogen permeation in Python

**Author:** Arseniy A. Kuzmin

## Abstract

This project describes several methods for solving the diffusion problem for hydrogen permeation through a metal membrane, starting with the simplest cases: one-dimensional membrane, no traps. Both explicit and implicit methods are considered, with the aim of selecting schemes that are fast and accurate. The numerical solution is implemented in [Python](https://www.python.org/). Calculations in pure Python are slow; common options to speed them up are to use [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/) for array and linear algebra operations, or [Numba](https://numba.pydata.org/) and JIT compilation for loops.

The explanation starts with explicit stencils, then implicit stencils (including Backward Euler as used in this package). Helpful references include the [numerical-mooc](https://github.com/numerical-mooc/numerical-mooc) Jupyter notebooks, and works by A. A. Pisarev and E. D. Marenkov (MEPhI) and S. K. Sharma.

## Contents

- [Diffusion equation](theory/diffusion.md) — permeation PDE and boundary conditions
- [Finite difference methods](theory/finite_methods.md) — Crank–Nicolson and grid setup
- [Equation cheat sheet](reference/equations.md) — stencils and \(\sigma\) definitions

## Figures

Figures generated from the notebooks are in the repository folder `figures/` (e.g. `figures/diffusionpyexample0.png`).
