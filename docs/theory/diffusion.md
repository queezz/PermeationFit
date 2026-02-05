# Permeation equation

The general equation for hydrogen permeation through a membrane (one-dimensional case):

\[
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + f(u)
\]

\[
\Gamma_{\mathrm{in}}(t) + D\frac{\partial u}{\partial x}\bigg|_{x = 0} - k_{u}u^{2}(0,t) = 0
\]

\[
-D\frac{\partial u}{\partial x}\bigg|_{x = L} - k_{d}u^{2}(L,t) = 0
\]

Here \(u\) is the concentration, \(t\) is time, \(x\) is the coordinate, \(D\) is the diffusion coefficient, \(k_u\) and \(k_d\) are the hydrogen recombination coefficients on the upstream and downstream sides, and \(f(u)\) may represent the contribution of hydrogen traps. The term \(\Gamma_{\mathrm{in}}(t)\) is the incident atomic hydrogen flux.

The matrix in the non-boundary points is the same as in the Crank–Nicolson formulation; the boundary coefficients are determined by the above boundary conditions. At the left boundary:

\[
\Gamma_{\mathrm{inc}}^n - k_u U_0^{n2} + D\frac{U_1^n - U_0^n}{\Delta x} = 0
\]

At the right boundary:

\[
k_d U_{J-1}^{n2} + D\frac{U_{J-1}^n - U_{J-2}^n}{\Delta x} = 0
\]

Solving both quadratic equations for \(U_0^n\) and \(U_{J-1}^n\) gives the boundary values from the previous time layer:

\[
U_0^{n+1} = -\frac{D}{2k_u\Delta x} + \frac{1}{2} \sqrt{\left(\frac{D}{k_u \Delta x}\right)^2 + \frac{4DU_1^n}{k_u \Delta x} + \frac{4\Gamma_{\mathrm{inc}}^n}{k_u}}
\]

\[
U_{J-1}^{n+1} = -\frac{D}{2k_d\Delta x} + \frac{1}{2} \sqrt{\left(\frac{D}{k_d \Delta x}\right)^2 + \frac{4DU_{J-2}^n}{k_d \Delta x}}
\]
