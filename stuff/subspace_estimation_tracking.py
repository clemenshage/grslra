import numpy as np
import grslra
from grslra.visualization import plot_lpnorm
from grslra.scaling import *
from matplotlib import pyplot as plt
from grslra.grst import grst

PROFILE = 1

if PROFILE:
    import cProfile

m = 100
n = 20000
relative_rank = 0.05

k = np.int(np.round(relative_rank*np.minimum(m, n)))
rho = 0.05

X_0, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rpca_lmafit(m, n, k, rho)

rate_Omega = 0.5
card_Omega = np.int(np.round(rate_Omega * m * n))
Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))

X = X_0[Omega]
#Omega = None
#X = X_0

scaling = Scaling_Percentile(67, 0.3) #TODO find out which scaling to use
#scaling = None

p = 0.5
mu_end = 1E-4

params_grpca = {"I_max": 2000, "PRINT": None, "samplesize": 1e4}
params_cg_y = {"I": 50}
params_cg_U = {"rho_start": 0.1}
params_cg_Y = {"rho_start": 0.1}
params_gd_U = {"t_init": 1, "rho_start": 0.9, "I": 1}
params_grst = {"init_factor": 0, "VERBOSE": 1, "PRINT": 1}

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

L = grst(X, k, p, mu_end, Omega=Omega, dimensions=(m, n), params=params_grst, params_grpca=params_grpca, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, params_gd_U=params_gd_U, params_cg_y=params_cg_y, L_0=L_0, scaling=scaling)

err = np.linalg.norm(L - L_0, ord='fro') / np.linalg.norm(L_0, ord='fro')
print "\n||L - L_0||_F / ||L_0||_F = ", '{:.8f}'.format(err)

if PROFILE:
    profile.disable()
    profile.dump_stats("profile.bin")