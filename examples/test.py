from permeation import BE, Parameters, step_G

Nt = 100
params = Parameters(
    Nx=30,
    Nt=Nt,
    T=1000.0,
    D=1.1e-8,
    L=2e-5,
    ku=1e-33,
    kd=2e-33,
    ks=1e18,
    G_generator=step_G(1.0, t_start_frac=0.0, t_end_frac=1.0 / 3),
)
# result = BE(params)

