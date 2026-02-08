# Permeation package map

High-level map of the refactored `permeation` package: what lives where, how data flows, and why it’s structured this way. For re-entering the codebase, not for API reference.

---

## Package layout (by role)

```
src/permeation/
├── physics/          # Forward model: PDE, parameters, G(t) shapes
├── inverse/          # Inverse problem: fit G(t) from measured pdp
├── viz/               # Plotting from result dicts only
├── utils.py           # Fitting metric (chi_square)
└── __init__.py        # Re-exports public API
```

**Group by role:** physics = “given G(t), what is pdp?”; inverse = “given pdp, what is G(t)?”; viz = “draw it”; utils = shared helpers.

---

## 1. Physics (`physics/`)

**Problem:** Solve 1D hydrogen diffusion through a membrane with recombination BCs. Given incident flux G(t), produce concentration and downstream flux (pdp).

**Owns:**

- Backward Euler time integration and spatial discretisation (`backward_euler.py`).
- Parameter container and default grid/constants (`materials.py`).
- G(t) generators: constant, step, multi-step, refinement helpers (`materials.py`).
- Public API: `BE(parameters, **kwargs)` and `Parameters` (`diffusion.py` re-exports).

**Does not:** Fit anything, read files, or know about “measurement” vs “model” grids. No inverse logic.

**Called by:** Inverse fit (calls `BE` and materials helpers). Users call `BE` directly for forward runs.

**Design:** Physics is a pure function of parameters; no global state. `BE` returns a dict (time, x, c, fluxes, pdp, …). G is passed in via parameters or overrides.

---

## 2. Inverse (`inverse/`)

**Problem:** Recover step values of G(t) from measured downstream pressure (pdp). Least-squares with optional L2 and total-variation regularization.

**Owns:**

- Mapping step values + step starts → G(t) → BE run → pdp on model grid (`inverse_fit.simulate_from_step_vals`).
- Interpolation of model pdp onto measurement grid (`interp_to_meas_grid`).
- Single-level fit (`fit_G_steps`) and multi-level zoom fit (`fit_G_steps_zoom`).
- Adaptive bin refinement (`fit_with_adaptive_bins`, `refine_bins_adaptive`).
- Workflow class that holds data + params and runs zoom fit (`workflow.InverseFitWorkflow`).

**Does not:** Implement the PDE (delegates to `physics.BE`). Does not own plotting (delegates to `viz`). Does not define Parameters (uses `physics.Parameters`).

**Called by:** Notebooks/scripts; workflow is the main entry for “load data → fit → plot”. Low-level routines are used by workflow or for custom pipelines.

**State/history:**

- **Zoom result** (`fit_G_steps_zoom` return value): `history` = one dict per zoom level (tstart, x_hat, pdp_hat, G_hat, cost, …). Optional `states` = every optimizer iteration (level, iter, x, cost) when `save_states=True`.
- **Workflow:** Holds `t_meas`, `pdp_meas`, `base_params`, optional truth data; after `fit()`, stores the zoom result in `_result` and exposes it via `.result`.

**Design:** Workflows are separated from solvers. `InverseFitWorkflow` is a thin wrapper: it keeps data and params, calls `fit_G_steps_zoom`, and delegates plotting to `viz`. All fit logic lives in `inverse_fit`; the workflow does not contain physics or solver code.

---

## 3. Visualization (`viz/`)

**Problem:** Produce figures from solver or inverse results. No physics, no fitting.

**Owns:**

- Plots that consume **solver output**: profiles, fluxes, concentration 3D, summary (`plotting.py`).
- Plots that consume **zoom result**: inverse summary (measured vs fitted pdp + G), zoom frame (one level), convergence history (cost vs level), export of zoom-state frames (`plotting.py`).
- Plot of G(t) and data (optionally with truth) for workflow before/after fit (`plot_G`).

**Does not:** Call the solver or the inverse. No physics constants or PDE logic. Operates only on dicts (e.g. `result` from `BE`, `zoom` from `fit_G_steps_zoom`).

**Called by:** Workflow’s `plot()` and `export_frames()` (which call into `viz`); users can also call plotting functions directly with a result dict.

**Design:** Visualization is decoupled: it takes data in, draws it. Physics is not embedded in plotting—no D, ks, or grid logic in `viz`; all of that lives in physics/inverse. Plotting is “result in → figure out”.

---

## 4. Utils (`utils.py`)

**Owns:** `chi_square(exp, calc)` — normalised sum of squared differences (scale by max(exp)). Used as a fitting cost when comparing calculated to experimental pdp.

**Does not:** Run solver or inverse; no I/O.

---

## Data flow (conceptual)

```
Experimental data (t_meas, pdp_meas)
         │
         ▼
   InverseFitWorkflow  (holds data + base_params)
         │
         ├──► fit() ──► fit_G_steps_zoom()
         │                    │
         │                    ├──► multi_step_G(steps) + BE(base_params)  [physics]
         │                    ├──► interp_to_meas_grid(model_pdp, t_meas)
         │                    └──► least_squares(residual, x0, …)  [per level]
         │
         │              ◄── zoom result dict (history, states?, tstart, x_hat)
         │
         ├──► .result  (stored for later)
         │
         └──► plot(kind) / export_frames()
                    │
                    └──► viz.plot_* (zoom, t_meas, pdp_meas, …)  →  figure/files
```

**Typical path:** Load or synthesize (t_meas, pdp_meas) and base_params → build workflow → `fit(n_levels, ...)` → zoom result is stored in workflow → `plot("summary")` or `plot("convergence")` or `export_frames()` use that result. Forward-only use: build `Parameters`, set G via materials helpers, call `BE(params)` → pass result dict to `plot_profiles`, `plot_fluxes`, etc.

---

## Where state/history lives

- **Zoom fit:** `fit_G_steps_zoom` returns a dict with `history` (per-level results) and optionally `states` (per-iteration when `save_states=True`). Workflow keeps this in `_result` after `fit()`.
- **Adaptive fit:** `fit_with_adaptive_bins` returns a dict with its own `history` (edges/x/scores per refinement). No workflow wrapper; caller holds the result.
- **Solver:** `BE` returns a single result dict (no accumulation across calls). Concentration history is inside that dict (e.g. `c`).

---

## Design decisions (why it’s like this)

- **Workflows vs solvers:** The workflow holds data and orchestration; the solver is “run BE with these params”. Fit algorithms live in `inverse_fit`; the workflow just calls them and stores the result. Easier to test solvers and fits in isolation and to reuse fit logic without the workflow.
- **Viz decoupled:** Plotting only sees result dicts. You can change how results are produced (e.g. different optimizer) without touching viz; you can swap viz (e.g. different backend) without touching physics or inverse.
- **Physics not in plotting:** All PDE and parameter knowledge stays in `physics/`. Viz never imports BE or Parameters. Avoids circular deps and keeps “what to draw” separate from “how it was computed”.
- **Single package, src layout:** One namespace (`permeation`), installable with `pip install -e .`. `docs/DECISIONS.md` records what was left in `oldcode/` (e.g. wavefit, file I/O) and why.

---

*This page is navigation and architecture only. For API details see the code and docstrings; for theory see the Theory and References sections.*
