from sys import argv
from grslra.scaling import *
import numpy as np
from grslra.tools import subspace_angle
from grslra.testdata import testdata_rpca_lmafit
from grslra.grpca import grpca
import time

# This experiment computes the phase transitions for incomplete observations (20% to 80% of missing entries)


# define scenarios
m = 200
n = 200
maxval = 0.6
step = 0.01

# estimate which scenarios cannot not be recovered, these will be skipped
impossible_thresh = 1.2

# Use 38th percentile as reference as there may be up to 60% large outliers
scaling = Scaling(percentile=38, val_at_percentile=0.17)

# high accuracy settings
p = 0.1
mu_end = 1E-8

params_grpca = {"PRINT": None, "VERBOSE": 0, "SMMPROD": False}
params_cg_Y = {}
params_cg_U = {}

values = np.arange(step, maxval + step, step)
nofvalues = values.size
nofscenarios = nofvalues ** 2 - 0.5 * ((2 - impossible_thresh) * nofvalues) ** 2

# initialize result array
result = np.zeros((nofvalues, nofvalues, 3))
result[:, :, 0] = 1.0
result[:, :, 1] = 90.0
result[:, :, 2] = 0

rate_Omega=float(argv[1])
card_Omega = np.int(np.round(rate_Omega * m * n))

# Estimate runtime of the evaluation
rho_min = step
k_min = np.int(np.round(step*np.minimum(m, n)))
t_start = time.time()
X, _, _, _,_ = testdata_rpca_lmafit(m, n, k_min, rho_min)
Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))
X = X[Omega]
grpca(X, k_min, p, mu_end, params=params_grpca, scaling=scaling, Omega=Omega, dimensions=(m,n))
t_min = time.time() - t_start

rho_max = maxval
k_max = np.int(np.round(maxval*np.minimum(m, n)))
t_start = time.time()
X, _, _, _,_ = testdata_rpca_lmafit(m, n, k_max, rho_max)
Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))
X = X[Omega]
grpca(X, k_min, p, mu_end, params=params_grpca, scaling=scaling, Omega=Omega, dimensions=(m,n))
t_max = time.time() - t_start

t_av = 0.5 * (t_max + t_min)
print "Estimated runtime: ", np.int(np.round(t_av * nofscenarios)), " seconds"


t_start = time.time()

for it_k_m in xrange(nofvalues):
    k_m = values[it_k_m]

    for it_rho in xrange(nofvalues):
        rho = values[it_rho]

        print "relative rank: ", k_m, "rho: ", rho

        if it_k_m + it_rho > impossible_thresh * nofvalues:
            result[it_k_m, it_rho, 0] = 1
            result[it_k_m, it_rho, 1] = 90
            result[it_k_m, it_rho, 1] = 0
            continue

        k = np.int(np.round(k_m * m))

        X_0, L_0, S_0, U_0, Y_0 = testdata_rpca_lmafit(m, n, k, rho)
        Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))
        X = X_0[Omega]

        t_startit = time.time()

        try:
            U, Y = grpca(X, k, p, mu_end, params=params_grpca, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, scaling=scaling, Omega=Omega, dimensions=(m, n))
            t = time.time() - t_startit
            L_p = np.dot(U,Y)
            err = np.linalg.norm(L_p - L_0, ord='fro') / np.linalg.norm(L_0, ord='fro')
            angle = subspace_angle(U, U_0)
            print "Error: ", err

            result[it_k_m, it_rho, 0] = err
            result[it_k_m, it_rho, 1] = angle
            result[it_k_m, it_rho, 2] = t
        except:
            pass

    np.savez('result_' + '{:d}'.format(np.int(100*rate_Omega)), result=result, maxval=maxval, step=step)

print "Finished in ", time.time()-t_start, " seconds"