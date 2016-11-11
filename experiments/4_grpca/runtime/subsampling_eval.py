import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import grslra
import json
import time
import cProfile

PROFILE = 0

k = 10
rho = 0.1
scaling = grslra.scaling.Scaling()

p = 0.1
mu_end = 1e-8

dims = np.round(np.sqrt([1e4, 2.5e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6, 5e6, 7.5e6, 1e7, 2.5e7, 5e7, 7.5e7, 1e8])).astype(np.int)

paramfiles=["default", "subsampling"]
nofexperiments = dims.__len__()
nofparamfiles = paramfiles.__len__()

rate_Omega = 1

times = np.zeros((nofexperiments, nofparamfiles))
errs_rmse = np.zeros((nofexperiments, nofparamfiles))

for i in xrange(nofexperiments):
    m = dims[i]
    n = m

    print "\n========================================================\nm = ", m, " Omega rate: ", rate_Omega

    X_0, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rpca_lmafit(m, n, k, rho)
    Omega = None
    X = X_0

    U_init, _ = np.linalg.qr(np.random.randn(m, k))

    for j in xrange(nofparamfiles):
        print "========================================================\nsettings: ", paramfiles[j]
        filename = paramfiles[j] + '.json'
        with open(filename, 'r') as f:
            params_grpca = json.load(f)
        for key, value in params_grpca.iteritems():
            if value == "None" or value == "none":
                params_grpca[key] = None
            if value == "False" or value == "false":
                params_grpca[key] = False
            if value == "True" or value == "true":
                params_grpca[key] = True

        params_grpca["VERBOSE"] = 0

        if PROFILE:
            profile = cProfile.Profile()
            profile.enable()

        t_start = time.time()

        U, Y = grslra.grpca(X, k, p, mu_end, params=params_grpca, Omega=Omega, dimensions=(m, n), scaling=scaling, U_0=U_0, Y_0=Y_0, U_init=U_init)

        t = time.time() - t_start

        if PROFILE:
            profile.disable()
            profile.dump_stats("profile_"  + '{:d}'.format(m) + "_" + paramfiles[j] + ".bin")


        times[i, j] = t
        print "\nFinished in ", t, " seconds"
        err_rmse = grslra.tools.rmse(U_0, Y_0, U, Y)
        errs_rmse[i, j] = err_rmse
        print "\n||L - L_0||_F / sqrt(mn) = ", '{:.8f}'.format(err_rmse)

        np.savez('result_subsampling.npz', times=times, errs_rmse=errs_rmse, dims=dims, paramfiles=paramfiles)