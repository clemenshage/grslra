import numpy as np
from optimization import CG
from problems import RobustSLRA
from spaces import Euclidean, GrassmannianSVD
from tools import orthogonality_check, parse_params, subspace_angle
from visualization import plot_matrices, plot_series

from structures import Hankel
from scipy.sparse import eye as speye
from scipy.sparse import vstack
from scipy.sparse import kron as spkron
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import svds
from scipy.sparse import lil_matrix


def grslra_batch(x, structure, k, p, mu, scaling=None, Omega=None, params=None, params_cg_U=None, params_cg_Y=None, U_init=None, Y_init=None, x_0=None, iteration=None):
    # this class implements the Grassmannian Robust Structured Low-Rank Approximation algorithm
    # Required inputs are the input data vector x, the structure, the rank estimate k as well as the cost function parameters p and mu (value at termination)
    # If data is incompletely observed also need to provide observation set Omega
    # by default, make a copy of the input data
    x = x.copy()
    if x_0 is not None:
        x_0 = x_0.copy()

    # parse parameters and set default parameters where necessary
    params = parse_params(params, 'grslra')
    params_cg_U = parse_params(params_cg_U, 'cg_U')
    params_cg_Y = parse_params(params_cg_Y, 'cg_Y')
    params_cg_U["VERBOSE"] = params["VERBOSE"]
    params_cg_Y["VERBOSE"] = params["VERBOSE"]

    if scaling is not None:
        x = scaling.scale_reference(x)
        if x_0 is not None:
            scaling.scale(x_0)

    grslra = RobustSLRA(x, structure, k, p, mu, params["rho_start"], Omega=Omega, PCA_INIT=params["PCA_INIT"], U_init=U_init, Y_init=Y_init)

    grassmannian = GrassmannianSVD(grslra.m, k)
    euclidean = Euclidean()
    minU_cg = CG(grslra, "U", grassmannian, params_cg_U)
    minY_cg = CG(grslra, "Y", euclidean, params_cg_Y)

    printit = 0
    i = 0

    # Outer loop performs the alternating minimization between L=UY and the Lagrangian Multiplier, until the structural constraints are sufficiently well fulfilled
    while i < params["I_max"] and grslra.rho <= params["rho_end"]:
        if params["VERBOSE"]:
            print "\nMinimization pass # ", i + 1, ", rho = ", grslra.rho
        ii = 0
        # Inner loop performs the alternating minimization between U and Y, until an intermediate solution L=UY is found
        while ii < params["II_max"]:
            if params["VERBOSE"] > 1:
                print "\tIteration # ", ii + 1

            U_old = grslra.U
            minU_cg.solve()

            # run a simple orthogonality check to test whether U is sufficiently orthogonal and orthogonalize if necessary
            grslra.U, R = orthogonality_check(grslra.U)
            if R is not None:
                grslra.Y = np.dot(R, grslra.Y)
            minY_cg.solve()

            angle_progress = subspace_angle(U_old, grslra.U)
            grslra.update_L()

            if params["VERBOSE"] > 1:
                print "\tangle_progress: ", angle_progress, "\n"

            if (i or ii) and angle_progress < params["delta"]:
                break
            ii += 1

        if params["PLOT"] > 1:
            plot_data(grslra, 1, visible=True)
            if x_0 is not None:
                plot_series(2, x_0, '-k', grslra.l, '-r', visible=True)
        if params["PRINT"] > 1:
            plot_data(grslra, 1, filename="/tmp/figures/matrices/" + '{:03d}'.format(printit) + ".png", visible=False)
            if x_0 is not None:
                plot_series(2, x_0, '-k', grslra.l, '-r',
                            filename="/tmp/figures/series/" + '{:03d}'.format(printit) + ".png", visible=False)
            printit += 1

        grslra.update_Lambda()

        # Increase rho depending on the number of iterations for the inner loop
        if ii > params["II_slow"]:
            grslra.rho *= params["c_rho_slow"]
        else:
            grslra.rho *= params["c_rho_fast"]
        i += 1

    if params["PLOT"]:
        plot_data(grslra, 1, visible=True)
        if x_0 is not None:
            plot_series(2, x_0, '-k', grslra.l, '-r', visible=True)
    if params["PRINT"]:
        if iteration is not None:
            plot_data(grslra, 1, filename="/tmp/figures/matrices/" + '{:03d}'.format(iteration) + ".png", visible=False)
            if x_0 is not None:
                plot_series(2, x_0, '-k', grslra.l, '-r', filename="/tmp/figures/series/" + '{:03d}'.format(iteration) + ".png", visible=False)
        else:
            plot_data(grslra, 1, filename="/tmp/figures/matrices.png", visible=False)
            if x_0 is not None:
                plot_series(2, x_0, '-k', grslra.l, '-r', filename="/tmp/figures/series.png", visible=False)

    l = grslra.l
    if scaling is not None:
        scaling.rescale(l)

        # firstly, rescale Y
        centering_backup = scaling.centering
        scaling.centering = False
        scaling.rescale(grslra.Y)
        # if a mean estimate has been subtracted the basis is augmented by the mean estimate
        if scaling.mu is not None:
            U_augmented = np.hstack((scaling.mu * np.ones((grslra.m, 1)), grslra.U))
            Y_augmented = np.vstack((np.ones((1, grslra.n)), grslra.Y))
            # re-orthogonalize
            grslra.U, R = orthogonality_check(U_augmented)
            if R is not None:
                grslra.Y = np.dot(R, Y_augmented)
        # reset centering status
        scaling.centering = centering_backup

    return l, grslra.U, grslra.Y


def plot_data(grslra, fignr, filename=None, visible=True):
    if grslra.x is None:
        x = np.zeros((grslra.N,))
        x[grslra.Omega] = grslra.x_Omega
        s = np.zeros((grslra.N,))
        s[grslra.Omega] = grslra.x_Omega - grslra.l[grslra.Omega]
    else:
        x = grslra.x
        s = np.abs(grslra.x - grslra.l)

    X = grslra.structure.struct_from_vec(x)
    S = grslra.structure.struct_from_vec(s)
    L = np.reshape(grslra.vec_L, (grslra.m, grslra.n), order='F')
    Lambda = np.reshape(grslra.vec_Lambda, (grslra.m, grslra.n), order='F')

    plot_matrices(fignr, (2, 3), X, "$\mathbf{X}$", (grslra.xmin, grslra.xmax),
                  L, "$\mathbf{L}$", (grslra.xmin, grslra.xmax),
                  S, "$\mathbf{X} - \mathbf{L}$", (0, grslra.xmax - grslra.xmin),
                  np.dot(grslra.U, grslra.U.T), "$\mathbf{U}\mathbf{U}^\top$", 0,
                  grslra.Y, "$\mathbf{Y}$", 0,
                  Lambda, "$\mathbf{\Lambda}$", 0,
                  filename=filename,
                  visible=visible
                  )


def slra_by_factorization(x, m, k, PRINT=0, x_0=None, Omega=None, N=None, PCA_INIT=False):
    if Omega is None:
        # assume x is the full data vector
        x = x
        N = x.size
        M_bar = None
    else:
        x_Omega = x
        if N is None:
            N = Omega.max()
        else:
            N = N
        x = np.zeros((N, ))
        x[Omega] = x_Omega

        entries = np.zeros((N,))
        entries[Omega] = 1.0
        M_bar = lil_matrix((N, N))
        M_bar.setdiag(entries)

    n = N - m + 1
    hankel = Hankel(m, n)

    if PCA_INIT:
        X_full = hankel.struct_from_vec(x)
        P, _, _ = svds(X_full, k)
    else:
        P, _ = np.linalg.qr(np.random.rand(m, k))

    L = None

    if M_bar is None:
        b = np.hstack((x, np.zeros((m * n,))))
    else:
        b = np.hstack((M_bar.dot(x), np.zeros((m * n,))))

    X = hankel.struct_from_vec(x)

    if x_0 is not None:
        X_0 = hankel.struct_from_vec(x_0)
    else:
        X_0 = None

    Pi_S_orth = speye(m * n) - hankel.Pi_S

    II_max = 1000
    I_max = 1000
    # Ishteva et al. propose to choose rho in the range (1, 1e14), but the values below work better in practice
    rho_start = 1e-5
    rho_end = 1e6
    c_rho_slow = 1.5
    c_rho_fast = 10

    rho = rho_start

    printit = 0
    II = 0

    for i in xrange(I_max):
        print "Iteration ", i, ", rho = ", rho
        for j in xrange(II_max):
            P_old = P
            if M_bar is None:
                M = vstack((hankel.S_pinv, np.sqrt(rho) * Pi_S_orth))
            else:
                M = vstack((M_bar.dot(hankel.S_pinv), np.sqrt(rho) * Pi_S_orth))

            A_L = M.dot(spkron(speye(n), P))
            L = np.reshape(lsqr(A_L, b)[0], (k, n), order='F')

            A_P = M.dot(spkron(L.T, speye(m)))
            P = np.reshape(lsqr(A_P, b)[0], (m, k), order='F')

            # Ishteva et al. propose to evaluate the change in the column space of P (such as the angle between subsequent subspaces), but it is not working well in practice
            # Instead, measure the relative difference between old and new estimate of P
            diff_P = np.linalg.norm(P - P_old, 'fro') / np.linalg.norm(P_old, 'fro')

            print diff_P
            if diff_P < 0.1:
                II = j
                break

        if II > 5:
            rho *= c_rho_slow
        else:
            rho *= c_rho_fast
        if rho >= rho_end:
            break
        if PRINT > 1:
            PL = np.dot(P, L)
            if X_0 is None:
                plot_matrices(1, (1, 3),
                              X, "$\mathbf{X}$", (X.min(), X.max()),
                              PL, "PL", (X.min(), X.max()),
                              X - PL, "$\mathbf{X} - \mathbf{P}\mathbf{L}$", (-X.max(), X.max()),
                              filename="/tmp/figures/matrices/slrabyF_" + '{:03d}'.format(printit) + ".png", visible=False)
            else:
                plot_matrices(1, (1, 4),
                              X, "$\mathbf{X}$", (X.min(), X.max()),
                              PL, "PL", (X.min(), X.max()),
                              X - PL, "$\mathbf{X} - \mathbf{P}\mathbf{L}$", (-X.max(), X.max()),
                              X_0 - PL, "$\mathbf{X}_0 - \mathbf{P}\mathbf{L}$", (-X.max(), X.max()),
                              filename="/tmp/figures/matrices/slrabyF_" + '{:03d}'.format(printit) + ".png", visible=False)
            printit += 1

    Y = np.dot(P, L)
    y = hankel.S_pinv.dot(Y.flatten('F'))
    return y
