import numpy as np
import grslra
from grslra.scaling import Scaling
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

X_1, L_1, S_1, U_1, Y_1 = grslra.testdata.testdata_rst_static(m, n, k, rho, omegasquared)
X_2, L_2, S_2, U_2, Y_2 = grslra.testdata.testdata_rst_static(m, n, k, rho, omegasquared)

Omega = None
X = np.hstack((X_1, X_2))
L_0 = np.hstack((L_1, L_2))

scaling = Scaling(percentile=38, val_at_percentile=0.17)

# determine scaling factor
scaling.scale_reference(X, Omega=Omega, dimensions=(m, n))
print "scaling: ", scaling.factor

p = float(argv[1])
mu_opt = (1-p) * (3 * np.sqrt(omegasquared) / scaling.factor) ** 2
print "mu_opt: ", mu_opt

mu = np.maximum(mu_opt, 0.1 * (3 * np.sqrt(omegasquared) / scaling.factor) ** 2)


params_grpca = {"PRINT": None}
params_cg_y = {"delta": 1e-8}
params_cg_U = {}
params_cg_Y = {}
params_gd_U = {"t_init": 1, "rho": 0.95, "I_max": 1}
params_grst = {"init_factor": 0, "VERBOSE": 2, "PRINT": None}

# params_grpca = {}
# params_cg_y = {}
# params_cg_U = {}
# params_cg_Y = {}
# params_gd_U = {}
# params_grst = {}


if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

L = grst(X, k, p, mu, params=params_grst, params_grpca=params_grpca, params_gd_U=params_gd_U, params_cg_y=params_cg_y, L_0=L_0, scaling=scaling)

err = np.linalg.norm(L - L_0, ord='fro') / np.linalg.norm(L_0, ord='fro')
print "\n||L - L_0||_F / ||L_0||_F = ", '{:.8f}'.format(err)

if PROFILE:
    profile.disable()
    profile.dump_stats("profile.bin")