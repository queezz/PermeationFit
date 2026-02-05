# Stencil cheat sheet

Space and time indices are denoted \(x\) and \(t\) here.

## Definition of \(\sigma\)

For **Crank–Nicolson**, the second derivative is approximated as the average of forward and backward Euler in time, giving:

\[
\frac{u_{x}^{t + 1} - u_{x}^{t}}{\Delta t} = \frac{D}{2 \Delta x^2}
\left(
(u_{x + 1}^{t + 1} - 2 u_{x}^{t + 1} + u_{x - 1}^{t + 1}) +
(u_{x + 1}^{t} - 2 u_{x}^{t} + u_{x - 1}^{t})
\right)
\]

So \(\sigma = \frac{D\Delta t}{2\Delta x^2}\).

For **explicit** (forward Euler) stencils:

\[
\frac{u_{x}^{t + 1} - u_{x}^{t}}{\Delta t} = \frac{D}{\Delta x^2}
\left(u_{x + 1}^{t} - 2 u_{x}^{t} + u_{x - 1}^{t}\right)
\]

In that case \(\sigma = \frac{D\Delta t}{\Delta x^2}\).

Below, \(\sigma = \frac{D\Delta t}{\Delta x^2}\) is used for all stencils.

## Stencils

**Forward Euler (explicit):**

\[
U_{x}^{t+1} = \sigma U_{x-1}^{t} + (1-2\sigma)U_{x}^{t} + \sigma U_{x+1}^{t}
\]

**Backward Euler (implicit):**

\[
-\sigma U_{x-1}^{t+1} + (1+2\sigma)U_{x}^{t+1} - \sigma U_{x+1}^{t+1} = U_{x}^{t}
\]

**Crank–Nicolson** (implicit, second order in time), with \(\sigma = \frac{D\Delta t}{\Delta x^2}\):

\[
-\frac{\sigma}{2} U_{x-1}^{t+1} + (1+\sigma)U_{x}^{t+1} - \frac{\sigma}{2} U_{x+1}^{t+1}
= \frac{\sigma}{2} U_{x-1}^{t} + (1-\sigma)U_{x}^{t} + \frac{\sigma}{2} U_{x+1}^{t}
\]

## Backward Euler with permeation boundary conditions

\[
-\sigma U_{x-1}^{t+1} + (1+2\sigma)U_{x}^{t+1} - \sigma U_{x+1}^{t+1} = U_{x}^{t}
\]

\[
U_{0}^{t+1} = -\frac{D}{2k_u\Delta x} + \frac{1}{2}\sqrt{\left(\frac{D}{k_u\Delta x}\right)^2 + 4\frac{D}{k_u\Delta x}U_{1}^{t+1} + \Gamma^{t+1}}
\]

\[
U_{L}^{t+1} = -\frac{D}{2k_d\Delta x} + \frac{1}{2}\sqrt{\left(\frac{D}{k_d\Delta x}\right)^2 + 4\frac{D}{k_d\Delta x}U_{L-1}^{t+1}}
\]
