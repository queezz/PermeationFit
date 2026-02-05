# Refactoring decisions and assumptions

## Proposed directory tree

```
PermeationFit/
├── README.md
├── pyproject.toml
├── mkdocs.yml
├── docs/
│   ├── index.md
│   ├── DECISIONS.md
│   ├── javascripts/
│   │   └── config.js
│   ├── theory/
│   │   ├── diffusion.md
│   │   └── finite_methods.md
│   └── reference/
│       └── equations.md
├── figures/
│   └── *.png
├── notebooks/
│   └── *.ipynb
├── oldcode/
│   ├── diffusion.py
│   ├── pdp_wavefit.py
│   └── pydiffuse.py
├── sections/          # original LaTeX (unchanged)
│   └── *.tex, *.pdf
├── src/
│   └── permeation/
│       ├── __init__.py
│       ├── diffusion.py
│       ├── solvers.py
│       ├── materials.py
│       └── utils.py
├── summary.tex
└── ...
```

## Package layout

- **src/permeation/** — Single package, `src` layout (PEP 517/518 style). Name `permeation` matches the problem domain.
- **diffusion.py** — Public API: `BE`, `parameters` (re-exported from materials/solvers).
- **solvers.py** — Backward Euler implementation only. Logic taken from `oldcode/diffusion.py` (canonical version; `pdp_wavefit.py` had the same BE plus `tools` and wavefit).
- **materials.py** — Default `parameters()` dict (grid, physics constants, flags).
- **utils.py** — `chi_square(exp, calc)` for fitting; no external `tools` dependency.

## What was not moved

- **pdp_wavefit.py** — `wavefit()`, `plot_result()`, and file I/O depend on external `tools` (paths, network drives, savitzky_golay, etc.). Left in `oldcode/`; notebooks or scripts can be updated later to use `permeation.BE` and `permeation.chi_square` with their own I/O.
- **pydiffuse.py** — Crank–Nicolson `run()` and PDF/plotting depend on `tools` and different BC handling. Not merged into the package to avoid scope creep; can be ported later if needed.
- **oldcode/** — Kept as-is for reference; no deletions.

## Behaviour and compatibility

- **BE()** — Signature accepts the same keys as before (Nx, Nt, T, D, L, ku, kd, ks, G, I, Uinit, PLOT, ncorrection, etc.). Extra keys (e.g. `Tend`, `saveU`) are absorbed by `**kwargs` and ignored. Return dict includes `fluxes` (DataFrame), `c`, `calctime`, and for compatibility `time` and `pdp`.
- **Windows factor** — The `ku`/`kd` × 2 correction on `os.name == "nt"` is preserved for TMAP7 agreement.
- **Bug fix** — The stray `u"Instead of u**2..."` string in the original was treated as code; it is now a comment (and the iterative correction loop is unchanged).

## Documentation

- **MkDocs Material** — `mkdocs.yml` and `docs/` with `index.md`, `theory/diffusion.md`, `theory/finite_methods.md`, `reference/equations.md`. LaTeX converted to Markdown with \( \) and \[ \] for math (MathJax).
- **Figures** — Not copied; referenced from repo root `figures/` in README. MkDocs site does not embed them; links point to repo paths.
- **summary.tex** — Content folded into `docs/index.md` (abstract + intro). Subfiles `sections/*.tex` became the theory and reference pages.

## Notebooks

- **notebooks/** — Unchanged except one new cell in `RunDiffusion.ipynb`: `from permeation import BE, parameters` and `import numpy as np`. Assumes the package is installed (`pip install -e .`) from the repo root. No conversion of notebooks to library code.

## Assumptions

- Python 3.8+.
- setuptools backend only; no PyPI/CI/tests added.
- Scientific results and numerical behaviour are preserved; only structure, packaging, and docs were changed.
