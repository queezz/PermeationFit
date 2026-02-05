# Permeation

Numerical solutions for **hydrogen permeation** through a one-dimensional metal membrane: diffusion PDE with recombination-type boundary conditions at the upstream and downstream surfaces.

## What it does

The package solves the 1D diffusion equation with flux/recombination BCs (as in e.g. TMAP7-style models). The main solver is **Backward Euler** (implicit) with iterative handling of the nonlinear boundary terms. You get time-dependent concentration in the membrane and inlet/outlet fluxes.

## Install

From the repository root:

```bash
pip install -e .
```

This installs the `permeation` package in editable mode. Dependencies: NumPy, SciPy, pandas.

## Quick start

```python
from permeation import BE, parameters

params = parameters()
params["Nt"] = 200
params["T"] = 3000
params["G"] = ...  # your incident flux (Nt+1)
result = BE(**params)
# result["fluxes"]: DataFrame with time, rel (inlet), perm (outlet)
# result["c"]: (Nt+1, Nx+1) concentration
# result["calctime"]: seconds
```

Notebooks in `notebooks/` (e.g. `RunDiffusion.ipynb`) use the same API after installing the package.

## Project layout

```
src/permeation/     # Python package
  __init__.py
  diffusion.py      # High-level API (BE, parameters)
  solvers.py        # Backward Euler implementation
  materials.py      # Default parameters
  utils.py          # e.g. chi_square for fitting
notebooks/          # Jupyter notebooks (exploration only)
docs/               # MkDocs Material site (theory + reference)
figures/            # Figures from notebooks
oldcode/            # Legacy scripts (not part of the package)
```

Documentation (theory and equation reference) is under `docs/`. Build with:

```bash
pip install mkdocs-material
mkdocs build
# or: mkdocs serve
```

## Non-goals

- No rewriting of numerical methods or physics.
- No CI, test suite, or PyPI publishing unless you add them later.
- Scientific results are unchanged; only layout and packaging are updated.

Refactoring notes and assumptions: **docs/DECISIONS.md**.
