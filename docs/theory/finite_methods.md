# Finite difference methods

This section is adapted from the [numerical-mooc](https://github.com/numerical-mooc/numerical-mooc) notebooks.

## The Crank–Nicolson method

The [Crank–Nicolson method](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method) is a finite difference scheme for the numerical integration of the heat equation and related PDEs. It is often used for reaction–diffusion systems in one space dimension:

\[
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + f(u), \qquad
\frac{\partial u}{\partial x}\bigg|_{x = 0, L} = 0
\]

Here \(u\) is the concentration, \(x\) is space, \(D\) is the diffusion coefficient, \(f\) is the reaction term, and \(L\) is the length of the domain. The boundary conditions are [Neumann](https://en.wikipedia.org/wiki/Neumann_boundary_condition) (zero flux at the boundaries).

## Grid and notation

Time and space are discretised as:

\[
t_n = n \Delta t, \quad n = 0, \ldots, N-1
\]

\[
x_j = j \Delta x, \quad j = 0, \ldots, J-1
\]

with \(\Delta t = T/N\) and \(\Delta x = L/J\). We write \(U_j^n = U(j\Delta x, n\Delta t)\) and define \(\sigma = \frac{D \Delta t}{2 \Delta x^2}\). The Crank–Nicolson scheme for the reaction–diffusion equation is:

\[
-\sigma U_{j-1}^{n+1} + (1+2\sigma) U_j^{n+1} -\sigma U_{j+1}^{n+1} = \sigma U_{j-1}^n + (1-2\sigma) U_j^n + \sigma U_{j+1}^n + \Delta t f(U_j^n)
\]

for \(j = 1,\ldots,J-2\). At the boundaries \(j=0\) and \(j=J-1\) the values \(U_{-1}^n\) and \(U_J^n\) are defined by the Neumann conditions (backward difference at \(j=0\), forward at \(j=J-1\)):

\[
\frac{U_1^n - U_0^n}{\Delta x} = 0, \qquad \frac{U_J^n - U_{J-1}^n}{\Delta x} = 0
\]

so \(U_0^n = U_1^n\) and \(U_{J-1}^n = U_J^n\). In vector form \(\mathbf{U}^n = (U_0^n, \ldots, U_{J-1}^n)^T\) the system is:

\[
A \mathbf{U}^{n+1} = B \mathbf{U}^n + \mathbf{f}^n
\]

and

\[
\mathbf{U}^{n+1} = A^{-1} \left( B \mathbf{U}^n + \mathbf{f}^n \right)
\]

Matrix \(A\) is constant and can be factorised once; \(B \mathbf{U}^n + \mathbf{f}^n\) is updated each time step.
