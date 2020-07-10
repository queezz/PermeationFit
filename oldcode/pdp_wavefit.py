import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import os

# import time

import tools


def BE(**kws):
    """
    Permeation solver. 
    """
    Nx = kws.get("Nx", 30)  # Number of space mesh points
    Nt = kws.get("Nt", 100)  # Number of time mesh points
    T = kws.get("T", None)  # Duration of the calculated permeation [s]
    D = kws.get("D", 1.1e-8)  # Diffusion coeff.
    I = kws.get("I", None)  #
    ku = kws.get("ku", None)  # upstream recombination coeff. []
    kd = kws.get("kd", None)  # downstream recombination coeff.
    ks = kws.get("ks", None)  # amplitude of the incident flux
    G = kws.get("G", None)  # initial guess for the incident flux
    L = kws.get("L", 2e-5)  # membrane thickness [m]
    PLOT = kws.get("PLOT", False)  # plot or not the results
    saveU = kws.get("saveU", None)  #
    Uinit = kws.get("Uinit", None)  # initial concentration profile
    # ------------------------------------------------------------------------------
    if os.name == "nt":
        # print 'windows'
        ku = 2 * ku
        kd = 2 * kd  # if this is applied, result is same with TMAP7.
    # ------------------------------------------------------------------------------
    G = G * ks
    if len(np.where(G < 0)[0]):
        return {
            "time": np.linspace(0, T, Nt + 1),
            "pdp": np.zeros(Nt + 1),
            "concentration": Uinit,
        }
    # start = time.clock()
    x = np.linspace(0, L, Nx + 1)  # mesh points in space
    t = np.linspace(0, T, Nt + 1)  # mesh points in time
    if I:
        u_1 = np.array([I(i) for i in x])  # initial concentration
    else:
        u_1 = np.copy(Uinit)
    # if len(np.where(U==0)[0]) > 0: print 'zeros'
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    F = D * dt / dx ** 2
    inlet = []
    outlet = []
    inlet.append(ku * u_1[0] ** 2)
    outlet.append(kd * u_1[Nx] ** 2)
    u = np.zeros(Nx + 1)
    if saveU:
        Usave = np.zeros((Nt + 1, Nx + 1), float)
    if saveU:
        Usave[0] = u_1
    if PLOT:
        plt.plot(x / 1e-6, u_1, "k-", lw=4)
    color_idx = np.linspace(0, 1, Nt)
    teta1 = D / (ku * dx)
    teta2 = D / (kd * dx)
    for n in range(0, Nt):
        # calculate u[1] and u[Nx-1] using explicit stencil
        g0 = F * u_1[0] + (1 - 2 * F) * u_1[1] + F * u_1[2]
        gL = F * u_1[Nx - 2] + (1 - 2 * F) * u_1[Nx - 1] + F * u_1[Nx]
        # put 1 for u[0] and u[Nx-1] in A
        A = diags(
            diagonals=[
                [0] + [-F for i in range(Nx - 1)],
                [1] + [1.0 + 2.0 * F for i in range(Nx - 1)] + [1],
                [-F for i in range(Nx - 1)] + [0],
            ],
            offsets=[1, 0, -1],
            shape=(Nx + 1, Nx + 1),
            format="csr",
        )
        # in the b (for BE) put roots of the quadratic equation for the border.
        b = np.array(
            [-teta1 / 2.0 + 0.5 * np.sqrt(teta1 ** 2 + 4 * teta1 * g0 + 4 * G[n] / ku)]
            + [i for i in u_1[1:Nx]]
            + [-teta2 / 2.0 + 0.5 * np.sqrt(teta2 ** 2 + 4 * teta2 * gL)]
        )
        # solve SLE
        u[:] = spsolve(A, b)
        u"Instead of u**2 put u*a, where a - u from previous step"
        for _ in range(3):
            a0 = u[0]
            aL = u[Nx]
            A = diags(
                diagonals=[
                    [-D / dx] + [-F for i in range(Nx - 1)],
                    [D / dx + ku * a0]
                    + [1.0 + 2.0 * F for i in range(Nx - 1)]
                    + [D / dx + kd * aL],
                    [-F for i in range(Nx - 1)] + [-D / dx],
                ],
                offsets=[1, 0, -1],
                shape=(Nx + 1, Nx + 1),
                format="csr",
            )
            b = np.array([G[n]] + [i for i in u_1[1:Nx]] + [0])
            u[:] = spsolve(A, b)
        u_1, u = u, u_1
        inlet.append(ku * u_1[0] ** 2)
        outlet.append(kd * u_1[Nx] ** 2)
        if PLOT:
            plt.plot(x / 1e-6, u_1, ".-", color=plt.cm.jet(color_idx[n]))
        if saveU:
            Usave[n + 1] = u_1

    if PLOT:
        font = {"family": "Times New Roman", "weight": "heavy", "size": 25}
        plt.rc("font", **font)
        plt.rcParams.update({"mathtext.default": "regular"})
        ax = plt.gca()
        ax.set_xlim(0, L / 1e-6)
        ax.set_xlabel("x ($\mu m$)", fontweight="heavy")
        ax.set_ylabel("concentration ($m^{-3}$)", fontweight="heavy")
        sf = os.path.join(tools.docs(0), "Stencils", "concentraion.png")  # %(F))
        plt.savefig(sf, dpi=300, bbox_inches="tight")
    # end = time.clock()
    result = dict()
    result.update(
        [("reflected", inlet), ("pdp", outlet), ("time", t), ("concentration", u_1)]
    )
    if saveU:
        return Usave, [t, outlet]
    else:
        return result


def chi(exp, calc):
    "Calculates chi square for given time series"
    exp = np.array(exp)
    calc = np.array(calc)
    M = exp.max()
    if M == 0:
        M = 1
    # print exp,calc,((exp-calc)**2/M**2).sum()
    return ((exp - calc) ** 2 / M ** 2).sum()


def parameters():
    u"""initial parameters for the fit"""
    Nx = 30
    Nt = 100
    T = 1000.0
    Tend = 705.0
    D = 1.1e-8
    L = 2e-5
    ku = 1e-33
    kd = 2e-33
    ks = 1e19
    PLOT = False
    Uinit = np.zeros(Nx + 1)
    G = np.zeros(Nt + 1)
    I = None
    kws = dict()
    kws.update(
        [
            ("Nx", Nx),
            ("Nt", Nt),
            ("T", T),
            ("D", D),
            ("Tend", Tend),
            ("L", L),
            ("I", I),
            ("G", G),
            ("ku", ku),
            ("kd", kd),
            ("ks", ks),
            ("PLOT", PLOT),
            ("Uinit", Uinit),
        ]
    )
    return kws


gpdp = list()


def wavefit(**kws):
    u"minimize using scipy"
    Nt = kws.pop("Nt", 100)
    T = float(kws.pop("T", 200))
    Tend = float(kws.pop("Tend", 100))
    probe = kws.pop("probe", "PDP4")
    shot = kws.pop("shot", 29733)
    ks = kws.pop("ks", 1.1e18)
    ku = kws.pop("ku", 1e-32)
    silent = kws.pop("silent", True)
    kd = kws.pop("kd", 1e-33)
    global gpdp

    def mbe(x, uinit=False):
        global gpdp
        try:
            x = x[0]
        except:
            pass
        G_stp = np.ones(Nt_stp) * x
        if x < 0:
            # flog.write('%.2e\t[%.2e %.2e]\t%e\t[%.4f]\t%e\n'%
            #     (ey[N*stp],gpdp[0],gpdp[1],x,G_stp[0],1e10))
            return 1e10
        pm.update(
            [
                ("G", G_stp),
                ("T", T_stp),
                ("Tend", Tend_stp),
                ("Nt", Nt_stp),
                ("Uinit", U),
            ]
        )
        res = BE(**pm)
        gpdp = res.pop("pdp")
        # print ey[N*stp],gpdp,x,G_stp,chi(ey[N*stp],gpdp)
        if not uinit:
            # flog.write('%.2e\t[%.2e %.2e]\t%e\t[%.4f]\t%e\n'%
            #     (ey[N*stp],gpdp[0],gpdp[1],x,G_stp[0],chi(ey[N*stp],gpdp)))
            pass
        if uinit:
            return res.pop("concentration")
        else:
            return chi(ey[N * stp], gpdp)

    # pth = os.path.join(docs(0),'Stencils')
    pth = os.path.join(tools.docs(0), "PDP")
    # data = np.loadtxt(os.path.join(pth,'expTest.txt'))
    # ex = data[:,0]; ey = data[:,1]
    PTH = tools.network_drives()
    fpth = os.path.join(
        PTH["freenas"], "Results", "%s" % probe, "%d_%s.txt" % (shot, probe)
    )
    if not os.path.exists(fpth):
        fpth = os.path.join(
            PTH["freenas"], "Results", "%s" % probe, "%d_%s.dat" % (shot, probe)
        )
    if not os.path.exists(fpth):
        fpth = os.path.join(tools.docs(0), "PyOut", "%d_%s.txt" % (shot, probe))
    print(fpth)
    data = np.loadtxt(fpth, skiprows=1)
    ex = data[:, 0]
    ey = data[:, 1]
    if ex[-1] < T:
        T = ex[-1]
    pm = parameters()
    Nx = pm.get("Nx")  # ;Nt=pm.get('Nt');T=pm.get('T');Tend=pm.get('Tend')
    U = np.zeros(Nx + 1)
    pm.update([("T", T), ("Tend", Tend), ("Nt", Nt)])
    tm = np.linspace(0, T, Nt + 1)
    dt = tm[1] - tm[0]
    pm.update([("ks", ks), ("ku", ku), ("kd", kd), ("Tend", Tend)])
    print(
        "#{} {} Tend={} T={} N={} ks={:.2e}, ku={:.2e}, kd={:.2e}".format(
            shot, probe, Tend, T, Nt, ks, ku, kd
        )
    )

    if Nt + 1 != len(ey):
        # print pm['Nt'],len(ey),ex[-1],pm['T']
        ef = interp1d(ex, ey)
        ex = tm
        ey = ef(tm)
    G = np.zeros(Nt + 1)
    stp = 1  # step in points for fitting time window, min=1, best resolution.
    # print int(Tend/T*Nt),Nt
    R = range(int(Tend / T * Nt) + int(Nt * 0.00))  # R = range(Nt)
    # clrs = np.linspace(0,1,len(R))
    # plt.plot(ex,ey,'k.--')
    Nt_stp = 1
    T_stp = dt
    Tend_stp = dt
    gpdp = []
    for N in R:
        # flog.write('{:.2%}\n'.format((N+1)/float(len(R))))
        # res = minimize(mbe,25,options={'maxiter':50})
        # res = minimize(mbe,25,method='CG')#Works well,but slow
        res = minimize(mbe, 25, method="Nelder-Mead")
        if not silent:
            print("{:.2%}\t{:.4f}".format((N + 1) / float(len(R)), res.x[0]))
        if res.x[0] < 0:
            G[N * stp] = 0
            U = mbe(0, True)
        else:
            G[N * stp - 1] = res.x[0]
            U = mbe(res.x[0], True)
        # flog.write('Ginc[%d] = %.4f\n'%(N*stp,res.x[0]))
        # plt.plot(tm_stp+N*stp*dt,gpdp,'-',color=plt.cm.jet(clrs[N]))
        # print gpdp[-1],ey[N*stp]

    G[-1] = 0
    pm.update(
        [("T", T), ("Tend", Tend), ("Nt", Nt), ("G", G), ("Uinit", np.zeros(Nx + 1))]
    )
    res = BE(**pm)
    t = res.pop("time")
    pdp = res.pop("pdp")
    with open(os.path.join(pth, "%d_%s.txt" % (shot, probe)), "w") as ff:
        ff.write("time\tGinc\tcalc\texp\n")
        for tg, ginc, clc, exp in zip(t, G * ks, pdp, ey):
            ff.write("%.4f\t%.4e\t%.4e\t%.4e\n" % (tg, ginc, clc, exp))
    plot_result(
        **{
            "time": t,
            "pdp": ey,
            "gfited": pdp,
            "ginc": G * ks,
            "title": "%d %s" % (shot, probe),
            "savefig": os.path.join(pth, "%d_%s.png" % (shot, probe)),
        }
    )
    plt.close()


def plot_result(**kws):
    u"plot function"
    gpdp = kws.pop("pdp", [0])
    ginc = kws.pop("ginc", [0])
    tm = kws.pop("time", [0])
    gfited = kws.pop("gfited", [0])
    title = kws.pop("title", "")
    savefig = kws.pop("savefig", os.path.join(tools.docs(0), "wave.png"))
    font = {"family": "Times New Roman", "weight": "heavy", "size": 18}
    plt.rc("font", **font)
    plt.rcParams.update({"mathtext.default": "regular"})
    fig = plt.figure(figsize=(7, 7), facecolor="w")
    gs = GridSpec(2, 1)
    gs.update(left=0.05, right=0.9, wspace=0.3, hspace=0.0)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    [ax.set_xticklabels([]) for ax in [ax1]]

    ax1.plot(tm, gpdp, "ko", label="experiment")
    ax1.plot(tm, gfited, "r.-", label="fit")
    ax2.plot(tm, ginc, "k.", label="sig")
    ginc_smth = tools.savitzky_golay(ginc, int(0.005 * len(ginc)) + 1, 1)
    ax2.plot(tm, ginc_smth, "r-", label="smooth")

    lbls = ["$\Gamma_{pdp}$", "$\Gamma_{inc}$"]
    [
        ax.set_ylabel(r"%s" % (lbl), fontweight="heavy")
        for ax, lbl in zip([ax1, ax2], lbls)
    ]
    [tools.grid_visual(ax) for ax in fig.axes]
    [tools.ticks_visual(ax) for ax in fig.axes]
    [ax.locator_params("y", nbins=5) for ax in [ax1, ax2]]
    ax2.set_xlabel("time (sec)", fontweight="heavy")
    [ax.get_yaxis().set_label_coords(-0.08, 0.5) for ax in fig.axes]
    ax1.legend(
        labelspacing=0.1,
        columnspacing=0.1,
        handletextpad=0.5,
        numpoints=1,
        framealpha=0,
        bbox_to_anchor=(0.7, 0.5),
    )
    [ax.yaxis.get_major_ticks()[-1].label1.set_visible(False) for ax in [ax2]]
    ax1.margins(0.1)
    ax2.margins(0.1)
    plt.suptitle(title)
    plt.savefig(savefig, dpi=300, bbox_inches="tight")

