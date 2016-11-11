import numpy as np
import grslra
from grslra.visualization import plot_lpnorm
from grslra.scaling import Scaling
from matplotlib import pyplot as plt
from grslra.grst import grst
from sys import argv

PROFILE = 0

if PROFILE:
    import cProfile

m = 50
n = 20000
k = 10
omegasquared = 1e-5

rate_Omega = 1.0
rho = 0.5

X_0, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rst_static(m, n, k, rho, omegasquared)

if rate_Omega < 1.0:
    card_Omega = np.int(np.round(rate_Omega * m * n))
    Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))
    X = X_0[Omega]
else:
    Omega = None
    X = X_0

scaling = Scaling(percentile=38, val_at_percentile=0.17)

# determine scaling factor
scaling.scale_reference(X, Omega=Omega, dimensions=(m, n))
print "scaling: ", scaling.factor


pvalues = [0.1, 0.4, 0.7, 1.0]

for i in xrange(pvalues.__len__()):
    p = pvalues[i]

    mu_opt = (1-p) * (3 * np.sqrt(omegasquared) / scaling.factor) ** 2
    print "mu_opt: ", mu_opt

    mu = np.maximum(mu_opt, 0.1 * (3 * np.sqrt(omegasquared) / scaling.factor) ** 2)

    params_grpca = {"PRINT": None}
    params_cg_y = {"delta": 1e-8}
    params_cg_U = {}
    params_cg_Y = {}
    params_gd_U = {"t_init": 1, "rho": 0.95, "I_max": 1}
    params_grst = {"init_factor": 0, "VERBOSE": 2, "PRINT": 0, "plot_step": 500}

    if PROFILE:
        profile = cProfile.Profile()
        profile.enable()

    L = grst(X, k, p, mu, Omega=Omega, dimensions=(m, n), params=params_grst, params_grpca=params_grpca, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, params_gd_U=params_gd_U, params_cg_y=params_cg_y, L_0=L_0, scaling=scaling, U_0=U_0)

    err = np.linalg.norm(L - L_0, ord='fro') / np.linalg.norm(L_0, ord='fro')
    print "\n||L - L_0||_F / ||L_0||_F = ", '{:.8f}'.format(err)

    if PROFILE:
        profile.disable()
        profile.dump_stats("profile.bin")