import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import grslra
import json
import time
import cProfile

PROFILE = 1

k = 10
rho = 0.1
rate_Omega = 0.1
scaling = grslra.scaling.Scaling()

p = 0.1
mu_end = 1e-8

dims = np.round(np.sqrt([1e6, 2.5e6, 5e6, 7.5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8, 5e8, 7.5e8, 1e9])).astype(np.int)

paramfiles=["default", "subsampling", "smmprod"]
nofexperiments = dims.__len__()
nofparamfiles = paramfiles.__len__()


times = np.zeros((nofexperiments, nofparamfiles))
errs_rmse = np.zeros((nofexperiments, nofparamfiles))

for i in xrange(nofexperiments):
    m = dims[i]
    n = m

    print "\n========================================================\nm = ", m, " Omega rate: ", rate_Omega

    if m <= 3000:
        card_Omega = np.int(np.round(rate_Omega * m * n))
        Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))
        X_0, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rpca_lmafit(m, n, k, rho)
        X = X_0[Omega]
    else:
        card_Omega = np.int(np.round(rate_Omega * m * n))
        Omega = (np.random.choice(m, card_Omega, replace=True), np.random.choice(n, card_Omega, replace=True))
        X, L_0, S_0, U_0, Y_0 = grslra.testdata.testdata_rpca_lmafit_bigdata(m, n, k, Omega, rho)

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

        U, Y = grslra.grpca(X, k, p, mu_end, params=params_grpca, Omega=Omega, dimensions=(m, n), scaling=scaling, U_init=U_init)

        t = time.time() - t_start

        if PROFILE:
            profile.disable()
            profile.dump_stats("profile_"  + '{:d}'.format(m) + "_" + paramfiles[j] + ".bin")


        times[i, j] = t
        print "\nFinished in ", t, " seconds"
        err_rmse = grslra.tools.rmse(U_0, Y_0, U, Y)
        errs_rmse[i, j] = err_rmse
        print "\n||L - L_0||_F / sqrt(mn) = ", '{:.8f}'.format(err_rmse)

        np.savez('result_smmprod.npz', times=times, errs_rmse=errs_rmse, dims=dims, rate_Omega=rate_Omega, paramfiles=paramfiles)