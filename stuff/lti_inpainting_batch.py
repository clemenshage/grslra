__author__ = 'cleha'
from grslra import testdata

from grslra.tools import inpainting_preprocess
from grslra.grslra_batch import grslra_batch
from grslra.structures import Hankel
from grslra.visualization import plot_lpnorm
from grslra.scaling import Scaling
import numpy as np

PROFILE = 1

if PROFILE:
    import cProfile

N = 50
firstmissing = 20
lastmissing = 21
m = 20
k = 5
n = N - m + 1

scaling = Scaling()

sigma=0.0

x, x_0, _, _ = testdata.testdata_lti_outliers(N, m, k, rho=0.1, amplitude=1, sigma=sigma)

# x +=100
# x_0 +=100

# determine scaling factor
scaling.scale_reference(x)
print "scaling: ", scaling.factor

p = 0.1
mu_opt = (1-p) * (3 * sigma / scaling.factor) ** 2
print "mu_opt: ", mu_opt
mu = np.maximum(mu_opt, 1e-4)

x_Omega, Omega = inpainting_preprocess(x, m, firstmissing, lastmissing)

hankel = Hankel(m, n)
grslra_params = {"VERBOSE": 2, "PRINT": 2, "II_slow": 10}

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

l, _, _ = grslra_batch(x_Omega, hankel, k, p, mu, params=grslra_params, x_0=x_0, Omega=Omega, scaling=scaling)

if PROFILE:
    profile.disable()
    profile.dump_stats("profile.bin")