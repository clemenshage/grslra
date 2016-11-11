from sys import argv
from grslra.scaling import Scaling
import numpy as np
from grslra.tools import subspace_angle
from grslra.testdata import testdata_rpca_ialm
from grslra.grpca import grpca
import time

# This experiment generates phase transitions using the data model by Candes et al. (2011)
# The cost function parameter p must be set as an inline parameter
# The parameters are chosen for fast processing and reasonably good accuracy


# define scenarios
m = 400
n = 400
maxval = 0.5
step = 0.01

# estimate which scenarios cannot not be recovered, these will be skipped
impossible_thresh = 1.2

scaling = Scaling(percentile=38, val_at_percentile=0.17)

p = float(argv[1])
mu_end = 1E-8

params_grpca = {"PRINT": None, "VERBOSE": 0, "SMMPROD": False, "c_mu": 0.5}
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

# Estimate run time
rho_min = step
k_min = np.int(np.round(step*np.minimum(m, n)))
t_start = time.time()
X, _, _, _,_ = testdata_rpca_ialm(m, n, k_min, rho_min)
grpca(X, k_min, p, mu_end, params=params_grpca, scaling=scaling)
t_min = time.time() - t_start

rho_max = maxval
k_max = np.int(np.round(maxval*np.minimum(m, n)))
t_start = time.time()
X, _, _, _,_ = testdata_rpca_ialm(m, n, k_max, rho_max)
grpca(X, k_max, p, mu_end, params=params_grpca, scaling=scaling)
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

        X, L_0, S_0, U_0, Y_0 = testdata_rpca_ialm(m, n, k, rho)

        t_startit = time.time()

        try:
            U, Y = grpca(X, k, p, mu_end, params=params_grpca, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, scaling=scaling)
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

    np.savez('result_' + '{:1.1f}'.format(p), result=result, maxval=maxval, step=step)

print "Finished in ", time.time()-t_start, " seconds"