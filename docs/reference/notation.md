# Numerical Diffusion — Notation & Vocabulary

This page collects the basic symbols and operators used in the numerical
solution of the diffusion equation, mainly in finite-difference form.

---

## Symbols

### \( u^n \)

Discrete solution vector at time step \( n \).

Represents the value of the field (e.g. temperature, concentration, density)
at all spatial grid points at time \( t = n\,\Delta t \).

---

### \( \Delta t \)

Time step size.

Controls the temporal resolution of the simulation.

---

### \( D \)

Diffusion coefficient.

May be a scalar (constant diffusion) or a function of space, time, or state.

---

## Operators and Matrices

### \( I \) — Identity matrix

Identity operator in discrete space.

Acts as:

$$
I\,u = u
$$

In matrix form, \( I \) has ones on the diagonal and zeros elsewhere.
It appears when time-derivative terms are written in operator form.

---

### \( L \) — Discrete Laplacian

Matrix representation of the spatial Laplacian operator \( \nabla^2 \).

Obtained by discretising second spatial derivatives using finite differences
(or finite volumes / finite elements).

In 1D (uniform grid), \( L \) is tridiagonal.

---

## Time Integration

### Backward Euler (implicit)

Time discretisation:

\[
\frac{u^{n+1} - u^n}{\Delta t} = D\,L\,u^{n+1}
\]

Leads to the linear system:

\[
\left( I - \Delta t\,D\,L \right) u^{n+1} = u^n
\]

Properties:
- Fully implicit
- First order in time
- Unconditionally stable for diffusion

---

## Derived Parameters

### \( r = \dfrac{D\,\Delta t}{\Delta x^2} \)

Dimensionless diffusion number.

Appears naturally in finite-difference discretisations of the Laplacian.

---

## Notes

- Bold symbols typically represent vectors or matrices.
- Operators become matrices after spatial discretisation.
- In finite-element formulations, \( I \) is replaced by a mass matrix.
