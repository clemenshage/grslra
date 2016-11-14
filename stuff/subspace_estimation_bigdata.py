import numpy as np
import grslra
from grslra import grpca
from grslra.visualization import plot_lpnorm
from grslra.scaling import *
import time

PROFILE = 1

if PROFILE:
    import cProfile

m = 50000
n = 50000

k = 10

rho = 0.05

rate_Omega = 0.002
card_Omega = np.int(np.round(rate_Omega * m * n))
Omega = (np.random.choice(m, card_Omega, replace=True), np.random.choice(n, card_Omega, replace=True))

X_0, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rpca_lmafit_bigdata(m, n, k, Omega, rho)

X=X_0

scaling = Scaling_Percentile(33, 0.1)
#scaling = None

p = 0.5
mu_start = 1E-1
mu_end = 1E-8
iterations = 1000

cg_params_Y = {"rho_0": 0.1, "rho_I": 0.9, "c": 1E-4, "thresh": 1E-4, "I": 100, "t_init": 1e10, "reset_rate": 10, "t_min": 1E-12}
cg_params_U = {"rho_0": 0.1, "rho_I": 0.9, "c": 1E-6, "thresh": 1E-4, "I": 100, "t_init": 1e10, "reset_rate": 10, "t_min": 1E-12}
grpca_params = {"p": 0.5, "mu_start": mu_start, "mu_end": mu_end, "c_mu": 0.5, "I": 1000, "thresh":1E-3, "cg_params_Y": cg_params_Y, "cg_params_U": cg_params_U, "PLOT": 0, "VERBOSE": 2, "scaling": scaling, "samplesize": 10000, "SMMPROD":True}

plot_lpnorm(grpca_params["p"], grpca_params["mu_start"], grpca_params["mu_end"])


if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

start = time.time()

U, Y = grpca(X, k, grpca_params, Omega=Omega, dimensions=(m, n), U_0=U_0, Y_0=Y_0)

stop = time.time()
print "Elapsed time: ", stop - start, " seconds"

if PROFILE:
    profile.disable()
    profile.dump_stats("profile.bin")