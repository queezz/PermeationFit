# Hydrogen permeation
Simple numerical diffusion models in Python.

This repository is a **small Python package plus examples** for solving the 1D hydrogen diffusion / permeation problem in a metal membrane.

It started as a LaTeX note collection and gradually turned into runnable code.  
Now it’s documented with **MkDocs** and focuses on **practical use + minimal theory**.

The goal here is not to be exhaustive or perfectly general, but to provide:

- working numerical schemes,
- clear assumptions,
- and examples that are easy to modify.

If you need a fast way to *run* a permeation model and understand what the code is doing, this is for you.

---

## What’s inside

- **1D diffusion equation** for hydrogen in a flat membrane  
  (no traps, no bulk reactions — the baseline case)

- **Finite-difference schemes**
  - explicit stencils (for intuition)
  - implicit schemes (Backward Euler, Crank–Nicolson–style grids)

- **Python implementations**
  - NumPy / SciPy for arrays and sparse solvers
  - structured so you can swap parameters and boundary conditions easily

- **Examples**
  - how to run a simulation
  - how to extract permeation flux
  - how numerical choices affect stability and speed

This is intentionally **example-driven**, not framework-heavy.

---

## What this is *not*

- Not a polished simulation framework
- Not a benchmark against every solver under the sun
- Not a review article

If you need traps, multiple species, or full thermo-kinetics — this code is meant to be a **starting point**, not the final word.

---

## Documentation map

- [Diffusion equation](theory/diffusion.md)  
  Governing PDE and boundary conditions used here

- [Finite difference methods](theory/finite-methods.md)  
  Grid setup, stencils, and stability notes

- [Equation cheat sheet](reference/equations.md)  
  Stencils, coefficients, and σ definitions used in the code

---

## References
This package does **not** attempt to reproduce or follow the full body of hydrogen
transport theory and simulation developed over the last decades.

Instead, it implements a **minimal recombination–diffusion model** as a clean,
transparent starting point — something that is easy to read, run, and modify.

That said, it sits within a much broader historical and methodological context,
which is worth being aware of: [A bit of history](reference/history.md). 


## Why MkDocs 
*(and not LaTeX)*

LaTeX was great for derivations.  
MkDocs is better for:

- code + explanation living together,
- quick edits,
- examples that actually run.

The math hasn’t disappeared — it’s just no longer pretending to be a paper.
