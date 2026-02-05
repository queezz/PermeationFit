# kws = parameters()
Nt = 100
params = {
    "Nx": 30,
    "Nt": Nt,
    "T": 1000.0,
    "D": 1.1e-8,
    "Tend": 300.0,  # only used in wavefit
    "L": 2e-5,
    "I": None,
    "ku": 1e-33,
    "kd": 2e-33,
    "ks": 1e18,
    "PLOT": False,
}
params["Uinit"] = np.zeros(params["Nx"] + 1)

params["G"] = np.zeros(params["Nt"] + 1)
params["G"][: int(Nt / 3)] = 1

