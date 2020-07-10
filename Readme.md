# Permeation simulation

Created 2020/07/10.



## What to do

The working solver is both in `diffusion.py` and in `pdp_wavefit.py`. But it uses both python loops and `scipy.sparse.linalg.spsolve`. This combination is slow, should be due to `for` loops. Adding `@jit` does not work, must rewrite `BE` function. Now main goal is to simplify it, and make it dimensionless. At the moment changing recombination coefficients leads to changes in convergence. Reducing them requires increased number of time steps for the solver to converge. 

