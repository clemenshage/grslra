import numpy as np
import grslra
from grslra.visualization import plot_lpnorm
from grslra import grpca
from grslra.scaling import *
from grslra.tools import load_params
import time

PROFILE = 1

if PROFILE:
    import cProfile

m = 400
n = 400
relative_rank = 0.5
k = np.int(np.round(relative_rank*np.minimum(m, n)))
rho = 0.01

#X_0, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rpca_lmafit(m, n, k, rho, sigma=1000000)
X_0, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rpca_ialm(m, n, k, rho)

# rate_Omega = 0.5
# card_Omega = np.int(np.round(rate_Omega * m * n))
# Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))
#
# X = X_0[Omega]
Omega = None
X = X_0

scaling = Scaling()

p = 0.1
mu_end = 1e-8

#plot_lpnorm(p, mu_start, mu_end)


#settings lmafit fast
# params_grpca = {"PRINT": None, "mu_start": mu_start, "rho_start_cg_U": 0.1, "rho_start_cg_Y": 0.1, "samplesize": 1e4, "VERBOSE": 1, "delta": 0.1}
# params_cg_Y = {"delta": 1e-4}
# params_cg_U = {"delta": 1e-4}

params_grpca = {"PRINT": 2, "VERBOSE": 1, "SMMPROD": False}#, "samplesize": None}# "rho_start_cg_U": 0.9, "rho_start_cg_Y": 0.9}
params_cg_Y = {}
params_cg_U = {}

# params_grpca = {"PRINT": None, "VERBOSE": 1, "samplesize": None, "rho_start_cg_U": 0.9, "rho_start_cg_Y": 0.9, "SMMPROD": False, "c_mu": 0.5}
# params_cg_Y = {"delta": 1e-6}
# params_cg_U = {"delta": 1e-6}



if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

t_start = time.time()

U, Y = grpca(X, k, p, mu_end, params=params_grpca, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, U_0=U_0, Y_0=Y_0, scaling=scaling)

print "Finished in ", time.time() - t_start, " seconds"

L = np.dot(U,Y)
err = np.linalg.norm(L - L_0, ord='fro') / np.linalg.norm(L_0, ord='fro')
print "\n||L - L_0||_F / ||L_0||_F = ", '{:.8f}'.format(err)

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.w_xaxis.gridlines.set_lw(3.0)
# ax.w_yaxis.gridlines.set_lw(3.0)
# ax.w_zaxis.gridlines.set_lw(3.0)
# ax.scatter(X[0], X[1], X[2], c='k', marker='o')
# ax.scatter(L_p[0], L_p[1], L_p[2], c='b', marker='o')
# ax.scatter(0, 0, 0, c='r', marker='o')
# ax.scatter(1,1,1,c='g', marker='o')
# plt.show()

if PROFILE:
    profile.disable()
    profile.dump_stats("profile.bin")