import numpy as np
import time
from grslra_batch import grslra_batch, slra_by_factorization
from tools import forecasting_preprocess, parse_params, subspace_angle
from visualization import plot_series


def grslra_online(x, structure, k, p, mu, N_f, params=None, params_cg_U=None, params_cg_Y=None, scaling=None, Omega=None, N=None):
    # This method realizes the online time series algorithm based on GRLSRA.
    # The algorithm slides over the input data and reuses previous estimates for U and Y in order to reduce the computational effort.
    # m defines the feature dimension, k defines the rank of the approximation and N_f sets the number of forecast samples

    m = structure.m
    U = None
    Y = None

    if Omega is None:
        N = x.shape[0]  # length of the input data vector
    elif N is not None:
        N = N
    else:
        N = Omega.max()  # if N is not provided then take the largest index of Omega

    grslra_params = parse_params(params, 'grslra')

    x_hat = np.zeros((N + N_f,))  # intialize output array

    j = 2 * m - 1
    J = N + 1

    angles = 90.0 * np.ones((J,))
    t = np.zeros((J, ))

    while j < J:
        print str(j) + "/" + str(N)
        ix_j = range(j - 2 * m + 1, j)

        if Omega is None:
            x_j = scaling.scale_reference(x[ix_j])
            x_j_Omega, Omega_j, n = forecasting_preprocess(x_j, m, N_f)
        else:
            Omega_ix_j = Omega[np.where(ix_j[0] <= Omega < ix_j[-1])]
            x_j_Omega = scaling.scale_reference(x[Omega_ix_j])
            Omega_j = Omega[Omega_ix_j].append(range(ix_j[-1], ix_j[-1] + N_f))

        t_start = time.time()
        if j == 2 * m - 1:  # initialize randomly at first iteration, then with previous subspace estimate
            l, U, Y = grslra_batch(x_j_Omega, structure, k, p, mu, Omega=Omega_j, iteration=j, params=grslra_params, params_cg_U=params_cg_U,  params_cg_Y=params_cg_Y, scaling=None)
        else:
            U_old = U
            l, U, Y = grslra_batch(x_j_Omega, structure, k, p, mu, Omega=Omega_j, U_init=U, Y_init=Y, iteration=j, params=grslra_params, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, scaling=None)
            angles[j] = subspace_angle(U_old, U)

        t[j] = time.time() - t_start
        print "Finished in ", t[j], " seconds"

        scaling.rescale(l)

        if j == 2 * m - 1:
            x_hat[: j + N_f] = l  # initially, store the full data vector of the low-rank output
        else:
            x_hat[j + N_f - 1] = l[-1]  # then only store the last entry (furthest predicted time instance)

        if grslra_params["PLOT"]:
            plot_series(2, x, 'k-', x_hat[:j + N_f - 1], 'r-', filename="auto", visible=True, xlim=(2*m-1, N + N_f))

        if grslra_params["PRINT"]:
            plot_series(2, x, 'k-', x_hat[:j + N_f], 'r-',
                        filename="/tmp/figures/series_online/" + '{:03d}'.format(j) + ".png", visible=False, xlim=(2*m-1, N + N_f))
        j += 1
        Y = np.roll(Y, -1, axis=1)
        Y[:, -1] = Y[:, -2]

    return x_hat, angles, t


def slra_by_factorization_online(x, m, k, N_f, Omega=None, N=None):
    if Omega is None:
        N = x.shape[0]  # length of the input data vector
    elif N is not None:
        N = N
    else:
        N = Omega.max()  # if N is not provided then take the largest index of Omega

    x_hat = np.zeros((N + N_f,))  # intialize output array

    j = 2 * m - 1
    J = N + 1

    t = np.zeros((J, ))

    while j < J:
        print str(j) + "/" + str(N)
        ix_j = range(j - 2 * m + 1, j)

        if Omega is None:
            x_j = x[ix_j]
            x_j_Omega, Omega_j, n = forecasting_preprocess(x_j, m, N_f)
        else:
            Omega_ix_j = Omega[np.where(ix_j[0] <= Omega < ix_j[-1])]
            x_j_Omega = x[Omega_ix_j]
            Omega_j = Omega[Omega_ix_j].append(range(ix_j[-1], ix_j[-1] + N_f))

        t_start = time.time()
        l = slra_by_factorization(x_j_Omega, m, k, Omega=Omega_j, N=2*m-1+N_f)
        t[j] = time.time() - t_start
        print "Finished in ", t[j], " seconds"

        if j == 2 * m - 1:
            x_hat[: j + N_f] = l  # initially, store the full data vector of the low-rank output
        else:
            x_hat[j + N_f - 1] = l[-1]  # then only store the last entry (furthest predicted time instance)

        j += 1

    return x_hat, t
