import numpy as np
from problems import RobustPCA
from spaces import Euclidean, GrassmannianSVD
from optimization import CG
from grslra.visualization import plot_matrices
from grslra.tools import subspace_angle, orthogonality_check, rmse
from grslra.tools import parse_params


def grpca(X, k, p, mu_end, params=None, Omega=None, dimensions=None, params_cg_U=None, params_cg_Y=None, scaling=None, L_0=None, U_0=None, Y_0=None, kappa=None, U_init=None, copy=True):
    # this function implements the Grassmannian Robust PCA algorithm
    # Required inputs are the input data X, the rank estimate k as well as the cost function parameters p and mu (value at termination)
    # If data is incompletely observed also need to provide observation set Omega and the dimensions of the data
    # by default, make copies of all input matrices
    if copy:
        X = X.copy()
        if L_0 is not None:
            L_0 = L_0.copy()
        if U_0 is not None:
            U_0 = U_0.copy()
        if Y_0 is not None:
            Y_0 = Y_0.copy()
        if U_init is not None:
            U_init = U_init.copy()

    # parse parameters and set default parameters where necessary
    params = parse_params(params, 'grpca')
    params_cg_U = parse_params(params_cg_U, 'cg_U')
    params_cg_Y = parse_params(params_cg_Y, 'cg_Y')
    params_cg_U["VERBOSE"] = params["VERBOSE"]
    params_cg_Y["VERBOSE"] = params["VERBOSE"]

    if L_0 is not None:
        params["CALC_L"] = True

    if scaling is not None:
        X = scaling.scale_reference(X, Omega=Omega, dimensions=dimensions)
        if L_0 is not None:
            scaling.scale(L_0)
        if Y_0 is not None and not scaling.centering:
            scaling.scale(Y_0)

    mu_levels = 1 + np.int(np.round(np.log(mu_end / params["mu_start"]) / np.log(params["c_mu"])))  # how many different values for mu will be evaluated during the shrinkage

    if dimensions is None:
        m, n = X.shape
    else:
        m, n = dimensions

    rpca = RobustPCA(X, k, p, params["mu_start"],  Omega=Omega, dimensions=dimensions, samplesize=params["samplesize"], SMMPROD=params["SMMPROD"],
                     PCA_INIT=params["PCA_INIT"], CALC_L=params["CALC_L"], U_init=U_init, kappa=kappa,)

    euclidean = Euclidean()
    grassmannian = GrassmannianSVD(m, k)

    minU_cg = CG(rpca, "U", grassmannian, params_cg_U)
    minY_cg = CG(rpca, "Y", euclidean, params_cg_Y)

    # check if step size shrinkage coefficient rho is initialized with a small value (say .1) and compute how it will be incremented to its final value (say .9)
    if params["rho_start_cg_U"] is not None and params["rho_start_cg_U"] != params_cg_U["rho"] and mu_levels > 1:
            minU_cg.rho = params["rho_start_cg_U"]
            inc_rho_U = (params_cg_U["rho"] - params["rho_start_cg_U"]) / np.float(mu_levels - 1)
    else:
        inc_rho_U = 0

    if params["rho_start_cg_Y"] is not None and params["rho_start_cg_Y"] != params_cg_Y["rho"] and mu_levels > 1:
            minY_cg.rho = params["rho_start_cg_Y"]
            inc_rho_Y = (params_cg_Y["rho"] - params["rho_start_cg_Y"]) / np.float(mu_levels - 1)
    else:
        inc_rho_Y = 0

    cost = 1000
    it = 0
    printit = 0
    i = 0

    errs_rmse = np.zeros((params["I_max"],))
    errs_rel = np.zeros((params["I_max"],))
    angles = np.zeros((params["I_max"],))

    for i in xrange(params["I_max"]):
        if params["VERBOSE"]:
            print "\nMinimization pass # ", i+1, ", mu = ", rpca.mu
        U_old = rpca.U.copy()
        Y_old = rpca.Y.copy()
        cost_old = cost
        minU_cg.solve()

        # run a simple orthogonality check to test whether U is sufficiently orthogonal and orthogonalize if necessary
        rpca.U, R = orthogonality_check(rpca.U)
        if R is not None:
            rpca.Y = np.dot(R, rpca.Y)
        cost = minY_cg.solve()  # solve for Y

        # check if cost is actual cost on all positision or just prediction on a subset. If not, compute actual cost. Then check for progress.
        if rpca.card_Psi != rpca.card_Omega:
            cost = rpca.get_full_cost()

        rel_progress = (cost_old-cost) / (cost_old + 1E-12)

        if rel_progress < 0:
            rpca.U = U_old
            rpca.Y = Y_old
            cost = cost_old
        else:
            if params["VERBOSE"]:
                rpca.print_cost(1)
                print "\tRelative progress: ", '{:.2e}'.format(rel_progress)

        # if no substantial progress is made, set new value for smoothing parameter
        if rel_progress < params["delta"]:
            it += 1
            if it == mu_levels:
                break
            rpca.mu = np.maximum(params["c_mu"] * rpca.mu, mu_end)
            if params["VERBOSE"]:
                print "Parameter mu changed to ", '{:.2e}'.format(rpca.mu)
            minU_cg.rho += inc_rho_U
            minY_cg.rho += inc_rho_Y
            cost = rpca.get_full_cost()

        rpca.update()

        if params["PLOT"] > 1:
            plot_data(rpca, L_0, 1, visible=True)

        if params["PRINT"] > 1:
            plot_data(rpca, L_0, 1, filename="/tmp/figures/matrices/" + '{:03d}'.format(printit) + ".png", visible=False)
            printit += 1

        # compute and store error measures depending on the available ground truth
        if L_0 is not None:
            L = np.dot(rpca.U, rpca.Y)
            err_rel = np.linalg.norm(L - L_0, ord='fro') / np.linalg.norm(L_0, ord='fro')
            errs_rel[i] = err_rel
            if params["VERBOSE"]:
                print "\n||L - L_0||_F / ||L_0||_F = ", '{:.8f}'.format(err_rel)

        if U_0 is not None:
            angle = subspace_angle(U_0, rpca.U)
            angles[i] = angle
            if params["VERBOSE"]:
                print "\nSubspace angle: ", '{:.1f}'.format(angle)

            if Y_0 is not None:
                err_rmse = rmse(U_0, Y_0, rpca.U, rpca.Y)
                errs_rmse[i] = err_rmse
                if params["VERBOSE"]:
                    print "\n||L - L_0||_F / sqrt(mn) = ", '{:.8f}'.format(err_rmse)

    if params["PLOT"]:
        plot_data(rpca, L_0, 1)

    if params["PRINT"]:
        plot_data(rpca, L_0, 1, filename="/tmp/figures/matrices/rpca.png", visible=False)

    if np.any(errs_rel) or np.any(errs_rmse) or np.any(angles):
        errs_rmse = errs_rmse[:i]
        errs_rel = errs_rel[:i]
        angles = angles[:i]
        # This file documents the progress in reducing the approximation error. Some evaluation scripts use the data for visualization.
        np.savez('errs.npz', errs_rmse=errs_rmse, errs_rel=errs_rel, angles=angles)

    if scaling is not None:
        # firstly, rescale Y
        centering_backup = scaling.centering
        scaling.centering = False
        scaling.rescale(rpca.Y)
        # if a mean estimate has been subtracted the basis is augmented by the mean estimate
        if scaling.mu is not None:
            U_augmented = np.hstack((scaling.mu, rpca.U))
            Y_augmented = np.vstack((np.ones((1, n)), rpca.Y))
            # re-orthogonalize
            rpca.U, R = orthogonality_check(U_augmented)
            if R is not None:
                rpca.Y = np.dot(R, Y_augmented)
        # reset centering status
        scaling.centering = centering_backup

        # return only the factors, not the full matrix
    return rpca.U, rpca.Y


def plot_data(rpca, L_0, fignr, filename=None, visible=True):
    L = np.dot(rpca.U, rpca.Y)
    if rpca.X is None:
        X = np.zeros((rpca.m, rpca.n))
        X[rpca.Omega] = rpca.X_Omega
    else:
        X = rpca.X
    if L_0 is not None:
        plot_matrices(fignr, (1, 4),
                      X, "$\mathbf{X}$", (rpca.Xmin, rpca.Xmax),
                      L_0 - L, "$\mathbf{L}_0 - \mathbf{L}$", (L_0.min(), L_0.max()),
                      L, "L", (L_0.min(), L_0.max()),
                      X - L, "$\mathbf{X} - \mathbf{L}$", (-rpca.Xmax, rpca.Xmax), filename=filename, visible=visible)
    else:
        plot_matrices(fignr, (1, 3),
                      X, "$\mathbf{X}$", (rpca.Xmin, rpca.Xmax),
                      L, "$\mathbf{L}$", (rpca.Xmin, rpca.Xmax),
                      X - L, "$\mathbf{X} - \mathbf{L}$", (-rpca.Xmax, rpca.Xmax),
                      filename=filename, visible=visible)
