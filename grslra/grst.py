import numpy as np
import time
from grpca import grpca
from optimization import CG, GDexact
from problems import RobustSubspaceTracking
from spaces import Euclidean, GrassmannianSVDrank1
from tools import parse_params, orthogonality_check, subspace_angle
from visualization import plot_series


def grst(X, k, p, mu, Omega=None, dimensions=None, params=None, params_grpca=None, params_cg_Y=None, params_cg_U=None, params_cg_y=None, params_gd_U=None, scaling=None, L_0=None, U_0=None, copy=True):
    # this function implements the Grassmannian Robust Subspace Tracking algorithm
    # Required inputs are the input data X, the rank estimate k as well as the cost function parameters p and mu
    # If data is incompletely observed also need to provide observation set Omega and the dimensions of the data
    # by default, make copies of all input matrices
    if copy:
        X = X.copy()
        if L_0 is not None:
            L_0 = L_0.copy()
        if U_0 is not None:
            U_0 = U_0.copy()

    # parse parameters and set default parameters where necessary
    params = parse_params(params, '../params/grst')
    params_grpca = parse_params(params_grpca, 'grpca')
    params_cg_U = parse_params(params_cg_U, 'cg_U')
    params_cg_Y = parse_params(params_cg_Y, 'cg_Y')
    params_gd_U = parse_params(params_gd_U, 'gd_U')
    params_cg_y = parse_params(params_cg_y, 'cg_y')
    params_cg_U["VERBOSE"] = params["VERBOSE"]
    params_cg_Y["VERBOSE"] = params["VERBOSE"]
    params_gd_U["VERBOSE"] = params["VERBOSE"]
    params_cg_y["VERBOSE"] = params["VERBOSE"]

    if dimensions is None:
        m, n = X.shape
    else:
        m, n = dimensions

    L = np.zeros((m, n))

    # Define the step size for plotting the error progression
    avstep = params["plot_step"]
    avpoints = n / avstep + 1

    errs_rel = np.zeros((n,))
    errs_rel_av = np.zeros((avpoints,))
    errs_rel_av[0] = 1.0
    angles = 90.0 * np.ones((n,))
    angles_av = np.zeros((avpoints, ))
    angles_av[0] = 90.0
    times = np.zeros((n, ))

    # The parameter init_factor defines the multiple of k number of samples that are used for batch initialization
    init_samples = params["init_factor"] * k

    if init_samples > 0:
        # Batch initialization phase
        if Omega is not None:
            ix_init = np.where(Omega[1] < init_samples)[0]
            Omega_init = (Omega[0][ix_init], Omega[1][ix_init])
            dimensions_init = (m, init_samples)
            X_init = X[ix_init]
        else:
            Omega_init = None
            dimensions_init = None
            X_init = X[:, : init_samples]

        if scaling is not None:
            X_init = scaling.scale_reference(X_init)
        if L_0 is not None:
            L_0_init = L_0[:, : init_samples]
            if scaling is not None:
                scaling.scale(L_0_init)
        else:
            L_0_init = None

        U_init, Y_init = grpca(X_init, k, p, mu, Omega=Omega_init, dimensions=dimensions_init, params=params_grpca,
                               params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, scaling=None, L_0=L_0_init)
        L_init = np.dot(U_init, Y_init)
        if scaling is not None:
            scaling.rescale(L_init)
        L[:, : init_samples] = L_init
    else:
        # No batch initialization
        U_init = None
        if scaling is not None:
            scaling.scale_reference(X)
            # Since no init sequence is defined, the whole sequence is used to determine the scaling parameters.
            # Could be replaced with a moving average estimation for real-world applications
            print "No init sequence for scale. Using a priori information about whole sequence for scaling"

    rst = RobustSubspaceTracking(m, k, p, mu, U_init=U_init)

    grassmannian = GrassmannianSVDrank1(m, k)
    euclidean = Euclidean()
    min_y_cg = CG(rst, "y", euclidean, params_cg_y)
    min_U_gd = GDexact(rst, "U", grassmannian, params_gd_U)

    for j in xrange(init_samples, n):
        if params["VERBOSE"]:
            print "\nProcessing sample # ", j, "/", n

        # Extract data sample, index set and ground truth and scale it
        if L_0 is not None:
            l_0 = L_0[:, j]

        if Omega is None:
            x_j = X[:, j]
            Omega_j = None
        else:
            ix_j = np.where(Omega[1] == j)[0]
            Omega_j = Omega[0][ix_j]
            x_j = X[ix_j].copy()

        if scaling is not None:
            scaling.scale(x_j)

        rst.load_sample(x_j, Omega_j)

        # y can be initialized in different ways

        # if Omega_j is None:
        #     y_init = np.dot(rst.U.T, x_j)
        # else:
        #     x_j_full = np.zeros((m, ))
        #     x_j_full[Omega_j] = x_j
        #     y_init = np.dot(rst.U.T, x_j_full)

        y_init = np.zeros(k,)

        t_start = time.time()
        min_y_cg.solve(initval=y_init)
        min_U_gd.solve()
        times[j] = time.time() - t_start

        # run a simple orthogonality check to test whether U is sufficiently orthogonal and orthogonalize if necessary
        rst.U, R = orthogonality_check(rst.U)
        if R is not None:
            rst.y = np.dot(R, rst.y)

        l = np.dot(rst.U, rst.y)
        if scaling is not None:
            scaling.rescale(l)
        L[:, j] = l

        if L_0 is not None:
            err_rel = np.linalg.norm(l - l_0) / np.linalg.norm(l_0)
            errs_rel[j] = err_rel
            if params["VERBOSE"]:
                print "err_rel = ", err_rel

            if j and not (j+1) % avstep:
                ix = (j+1) / avstep
                errs_rel_av[ix] = np.mean(errs_rel[j - avstep+1:j])
                if params["PRINT"]:
                    plot_series(11, errs_rel_av, '-k', filename='/tmp/figures/tracking/err_' + '{:05d}'.format(j + 1) + '.png', visible=False)

        if U_0 is not None:
            angle = subspace_angle(U_0, rst.U)
            angles[j] = angle
            if params["VERBOSE"]:
                print "err_sub: ", '{:1.1f}'.format(angle), "\n"

            if j and not (j+1) % avstep:
                ix = (j+1) / avstep
                angles_av[ix] = np.mean(angles[j - avstep + 1:j])
                if params["PRINT"]:
                    plot_series(12, angles_av, '-k', filename='/tmp/figures/tracking/angle_' + '{:05d}'.format(j + 1) + '.png', visible=False)

    if np.any(errs_rel) or np.any(angles):
        np.savez('errs' + '_k_{:d}'.format(k) + '_p_{:1.1f}'.format(p) + '.npz', errs_rel=errs_rel, angles=angles, errs_rel_av=errs_rel_av, angles_av=angles_av, times=times, avstep=avstep)
    return L