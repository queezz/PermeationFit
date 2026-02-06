# Backward Euler diffusion solver (BE)

This page documents the numerical method used in the **Backward Euler (BE)**
permeation solver implemented in this project.  
It serves as a **reference anchor** for the solver design, terminology, and
implementation choices.

Derivations, stencil diagrams, and visualizations will be added later.

---

## Time integration

### Backward Euler

- **Type:** fully implicit, first-order in time
- **Used for:** time integration of the diffusion equation

The diffusion equation is discretised in time as:

\[
\frac{u^{n+1} - u^n}{\Delta t}
=
D \frac{\partial^2 u^{n+1}}{\partial x^2}
\]

This leads to a linear system of the form:

\[
(I - \Delta t\,D\,L)\,u^{n+1} = u^n
\]

where \(L\) is the discrete Laplacian operator, and \(I\) is identity operatro, see [notation](../reference/notation.md#operators-and-matrices)

**References**

- [Backward Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method){ target=_blank }
- [Implicit methods](https://en.wikipedia.org/wiki/Implicit_method){ target=_blank }

---

## Spatial discretisation

### Central finite differences

- **Grid:** uniform, one-dimensional
- **Order:** second-order accurate in space
- **Operator:** three-point stencil for the second derivative

Used to discretise the Laplacian in the diffusion equation.

**References**

- [Finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method){ target=_blank }
- [Discrete Laplacian](https://en.wikipedia.org/wiki/Discrete_Laplacian){ target=_blank }

---

## Linear system solution

### Sparse tridiagonal system

Each time step requires solving a sparse tridiagonal linear system resulting
from the implicit diffusion operator.

- Matrix assembly: `scipy.sparse.diags`
- Linear solver: `scipy.sparse.linalg.spsolve`

**References**

- [SciPy sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html){ target=_blank }
- [Tridiagonal matrix algorithm](https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm){ target=_blank }

---

## Boundary conditions

### Nonlinear recombination boundary conditions

At the upstream and downstream boundaries, the flux depends quadratically on
the local concentration:

\[
\Gamma \propto u^2
\]

This introduces nonlinearity only at the boundaries; the interior diffusion
equation remains linear.

---

## Nonlinear treatment

### Picard (fixed-point) iteration

The nonlinear boundary terms are handled using Picard iteration:

\[
u^2 \;\approx\; a \cdot u
\]

where the coefficient \(a\) is taken from the previous iterate.  
The resulting linear system is solved repeatedly until convergence.

An explicit predictor is used only to initialise the boundary values before
iteration.

**References**

- [Picard iteration](https://en.wikipedia.org/wiki/Picard_iteration){ target=_blank }
- [Fixed-point iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration){ target=_blank }
- [Picard's iteration method](https://adamdjellouli.com/articles/numerical_methods/7_ordinary_differential_equations/picards_method){ target=_blank}

---

## Notes

- The scheme is unconditionally stable with respect to the diffusion CFL
  condition.
- Numerical damping of fast transients is expected (typical for Backward Euler).
- Nonlinearity is confined to the boundary treatment.

Stencil visualizations and convergence illustrations will be added later.
