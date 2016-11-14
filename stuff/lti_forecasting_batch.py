import grslra
from grslra import testdata
from grslra.grslra_batch import grslra_batch
from grslra.structures import Hankel
from grslra.scaling import *
from grslra.tools import forecasting_preprocess

PROFILE = None
PLOT = None
VERBOSE = 1
PRINT = 2

if PROFILE:
    import cProfile

N = 200
r = 50

m = 50
k = 10
n = N - m + 1

scaling = Scaling()

sigma=0.1

x, _, x_0, X_0, _, _ = testdata.testdata_lti_outliers(N, m, k, rho=0.1, amplitude=5, sigma=sigma)

# determine scaling factor
scaling.scale_reference(x.copy())
print "scaling: ", scaling.factor

p = 0.1
mu_opt = (1-p) * (3 * sigma / scaling.factor) ** 2
print "mu_opt: ", mu_opt
mu = np.maximum(mu_opt, 1e-4)

x_Omega, Omega, n = forecasting_preprocess

hankel = Hankel(m, n)

grslra_params = {"PRINT": 2, "VERBOSE": 1}

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

l, _, _ = grslra_batch(x, hankel, k, p, mu, params=grslra_params, x_0=x_0, Omega=Omega, scaling=scaling)

if PROFILE:
    profile.disable()
    profile.dump_stats("profile.bin")