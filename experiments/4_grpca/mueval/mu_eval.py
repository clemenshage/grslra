import numpy as np
import grslra
from grslra import grpca
from grslra.scaling import Scaling
from numpy import random as npr

# This experiment evaluates the proposed heuristic for the selection of the smoothing parameter mu
# The assumed optimum value of mu is computed and the progress in reducing the residual error is evaluated for varying values of mu

m = 200
n = 200
relative_rank = 0.1

k = np.int(np.round(relative_rank*np.minimum(m, n)))
rho = 0.1
sigma = 0.05

scaling = Scaling()

X_0, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rpca_lmafit(m, n, k, rho)

N = sigma * npr.randn(m, n)
X = X_0 + N

# determine scaling factor
scaling.scale_reference(X_0)
print "scaling: ", scaling.factor

p = 0.1
mu_opt = (1-p) * (3 * sigma / scaling.factor) ** 2
print "mu_opt: ", mu_opt

params_grpca = {"PRINT": None, "rho_start_cg_U": 0.9, "rho_start_cg_Y": 0.9, "samplesize": None, "VERBOSE": 1, "delta": 1e-4, "c_mu": 0.5}
params_cg_Y = {"delta": 1e-8}
params_cg_U = {"delta": 1e-8}


mus = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
mulabels= ["10^{-6}", "10^{-5}", "10^{-4}", "10^{-3}", "10^{-2}"]

errs_rel = []

for i in xrange(mus.__len__()):
    mu_end = mus[i]
    _, _ = grpca(X, k, p, mu_end, params=params_grpca, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, L_0=L_0, scaling=scaling)
    data = np.load('errs.npz')
    errs_rel += [data["errs_rel"]]

np.savez('errs_all.npz', errs_rel=errs_rel, mus=mus, mulabels=mulabels)