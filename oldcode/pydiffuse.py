# -*- coding: utf-8 -*-
# pyfit
from matplotlib.backends.backend_pdf import PdfPages

u"""
pyfit: diffusion calculation only with Python.
Crank-Nicolson stencil, numpy.linalg.solve() Gaussian method to solve tridiagonal matrix equation on each time layer.
"""
import numpy as np
import matplotlib.pylab as plt
import os
from scipy.interpolate import interp1d
import tools as tl


def ltexp(value, decplace=1):
    exponent = int(np.floor(np.log10(abs(value))))
    coeff = round(value / np.float(10 ** exponent), decplace)
    return r"%s\times 10^{%d}" % (coeff, exponent)


def run(g_inc, L=20e-6, T=900, dt=1, J=40, k_u=1e-33, k_d=1e-30, suffix="a", D=1e-9):
    u""" T = time, dt - time step, J - x steps, k_u, k_d - recombination """
    dx = float(L) / float(J - 1)  # Grid parameter
    x_grid = np.array([j * dx for j in range(J)])  # Grid
    N = 1 + int(float(T) / float(dt))  # Time step
    t_grid = np.array([n * dt for n in range(N)])  # Time grid
    # plt.plot(t_grid,'.')
    # plt.show()
    T_membrane = 573.0
    # D=2.9e-7*np.exp(-0.23*1.6e-19/(1.38e-23*T_membrane)) # Diffusion coeffitient for U
    sigma = float(D * dt) / float((2.0 * dx * dx))
    if 0:
        print("%.3e:\tN" % N)
        print("%.3e:\tdx" % dx)
        print("%.3e:\tsigma" % sigma)
        print("%.3e:\tdt" % dt)
    # suffix = '%.0es %.0e'%(dt, dx)

    u""" initial concentration """
    U = np.array([0.0 for _ in range(J)])
    ff = interp1d(g_inc[:, 0], g_inc[:, 1])
    g_inc = ff(t_grid)
    if 0:
        plt.plot(t_grid, g_inc, ".")
        plt.margins(0.1)
        plt.show()

    def f_vec(U, ti):
        "ti - time index"
        upstream = -D / (2.0 * k_u * dx) + 0.5 * np.sqrt(
            (D / (k_u * dx)) ** 2 + 4 * D * U[1] / (k_u * dx) + 4 * g_inc[ti - 1] / k_u
        )
        downstream = -D / (2.0 * k_d * dx) + 0.5 * np.sqrt(
            (D / (k_d * dx)) ** 2 + 4 * D * U[-2] / (k_d * dx)
        )
        vivec = np.array([0.0 for _ in range(J)])
        vivec[0] = upstream
        vivec[-1] = downstream
        return vivec

    plt.gca().margins(0.1)

    u""" Matrixes """
    A = (
        np.diagflat([-sigma for _ in range(J - 1 - 1)] + [0.0], -1)
        + np.diagflat([1.0] + [1.0 + 2.0 * sigma for _ in range(J - 2)] + [1.0])
        + np.diagflat([0.0] + [-sigma for _ in range(J - 1 - 1)], 1)
    )

    B = (
        np.diagflat([sigma for _ in range(J - 1 - 1)] + [0.0], -1)
        + np.diagflat([0.0] + [1.0 - 2.0 * sigma for _ in range(J - 2)] + [0.0])
        + np.diagflat([0.0] + [sigma for _ in range(J - 1 - 1)], 1)
    )

    U_record = []
    U_record.append(U)
    # U_record.append([U[0],U[-1]])
    u""" Solving matrix equation for all time-layers"""
    for ti in range(1, N):
        U_new = np.linalg.solve(A, B.dot(U) + f_vec(U, ti))  # numpy Gauss
        U = U_new
        U_record.append(U)
        # U_record.append([U[0],U[-1]])
    fig = plt.figure()
    if 1:
        ax2 = fig.add_subplot(2, 1, 2)
        plt.xlabel("x")
        plt.ylabel("concentration")
        nn = 20
        tt = np.linspace(0 + T / float(nn), T - T / float(nn), nn)
        plt.plot(x_grid, U_record[0], "k.", label="t = %.1f" % t_grid[0], lw=2)
        color_idx = np.linspace(0, 1, nn)
        for ij, t1 in enumerate(tt):
            plt.plot(
                x_grid,
                U_record[int(t1 / dt)],
                label="t = %.2f" % t_grid[int(t1 / dt)],
                color=plt.cm.jet(color_idx[ij]),
            )  # @UndefinedVariable

        plt.plot(x_grid, U, "k--", label="t = %.1f" % t_grid[-1], lw=2)
        # legend(framealpha = 0.8)
        plt.margins(0.1)
        pth = os.path.join(tl.docs(0), "pyfit", "diff_profile%s.png" % suffix)
        dirpth = os.path.dirname(pth)
        if not os.path.isdir(dirpth):
            os.makedirs(dirpth)
        # plt.savefig(pth,bbox_inches = 'tight', dpi = 300)
    ax2 = fig.add_subplot(2, 1, 1)
    U_r = np.array(U_record)
    try:
        plt.plot(t_grid, U_r[:, -1] ** 2 * k_d, label="out")
    except:
        plt.plot(t_grid, U_r[1] ** 2 * k_d, label="out")
    # plot(t_grid,U_r[:,0]**2*k_u,label = 'input')
    if 0:
        "plot 24248 PDP6 experimental result"
        epth = os.path.join(
            tl.docs(5), "workspace", "fit", "output", "24248_PDP6_gamma_pdp_exp.txt"
        )
        expdata = np.loadtxt(epth, skiprows=1)
        plt.plot(expdata[:, 0], expdata[:, 1], label="29005 PDP6")
    plt.legend(framealpha=0.8)
    plt.margins(0.1)
    pth = os.path.join(tl.docs(0), "pyfit", "diff_fluxes%s.png" % suffix)
    plt.suptitle(
        r"$%ss\;%sm\;k_u = %s;k_d = %s;D = %s$"
        % (ltexp(dt, 0), ltexp(dx, 1), ltexp(k_u, 1), ltexp(k_d, 1), ltexp(D, 2))
    )
    plt.grid(True)
    return plt.gcf()


def scan_parameters():
    """change ku,kd, D and save all results in multipage PDF file for comparison"""
    g_inc = np.array([[-1, 0], [0, 0], [1, 5e18], [900, 5e18], [901, 0], [1101, 0]])

    # dT =[1,0.1,0.01]
    # JJ = [5,10,20,30,40,50]
    pth = os.path.join(tl.docs(0), "pyfit", "D.pdf")
    dirpth = os.path.dirname(pth)
    if not os.path.isdir(dirpth):
        os.makedirs(dirpth)
    i = 0
    ru = np.linspace(5e-34, 5e-32, 30)
    ru = [1e-33]
    rd = [1e-33]
    rD = np.linspace(5e-10, 5e-8, 30)
    # rD = [2.7e-9]
    with PdfPages(pth) as pdf:
        for D in rD:
            for ku in ru:
                for kd in rd:
                    i += 1
                    print("{:.1%}".format(float(i) / (len(ru) * len(rd) * len(rD))))
                    fig = run(
                        g_inc,
                        L=20e-6,
                        T=1100,
                        dt=0.01,
                        J=20,
                        k_u=ku,
                        k_d=kd,
                        suffix="ku%.3e" % ku,
                        D=D,
                    )
                    pdf.savefig(fig)
                    plt.close()


def main():
    g_inc = np.array([[-1, 0], [0, 0], [1, 5e18], [900, 5e18], [901, 0], [1101, 0]])
    pth = os.path.join(tl.docs(0), "pyfit", "points.pdf")
    dirpth = os.path.dirname(pth)
    if not os.path.isdir(dirpth):
        os.makedirs(dirpth)
    i = 0
    dT = [0.01, 0.001, 0.0005]
    JJ = [2, 3, 4, 5, 10, 20, 30, 40, 50]
    JJ = [4]
    ku = 1e-33
    kd = 1e-33
    D = 2.7e-9
    with PdfPages(pth) as pdf:
        for dt in dT:
            for xpoints in JJ:
                i += 1
                print("{:.1%}".format(float(i) / (len(dT) * len(JJ))))
                fig = run(
                    g_inc,
                    L=20e-6,
                    T=1100,
                    dt=dt,
                    J=xpoints,
                    k_u=ku,
                    k_d=kd,
                    suffix="ku%.3e" % ku,
                    D=D,
                )
                pdf.savefig(fig)
                plt.close()

    pass


if __name__ == "__main__":
    main()
