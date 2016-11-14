__author__ = 'cleha'

from grslra.structures import Hankel
from grslra import testdata, grslra_online
from grslra.visualization import plot_lpnorm
from grslra.scaling import *
import matplotlib
import cProfile
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})

scaling = Scaling()

sigma=0.01

N = 100
m = 20
k = 5
n = N - m + 1
r = 5

x, _, _, _, _, _ = testdata.testdata_lti_outliers(N, m, k, rho=0.0, amplitude=1, sigma=sigma)

# determine scaling factor
scaling.scale_reference(x.copy())
print "scaling: ", scaling.factor

p = 0.1
mu_opt = (1-p) * (3 * sigma / scaling.factor) ** 2
print "mu_opt: ", mu_opt
mu = np.maximum(mu_opt, 1e-4)

# grslra_params = {"rho_start": 1e-2, "rho_end": 1e2, "min_progress": 0.01, "VERBOSE": 1, "PRINT": 1}

grslra_params = {"VERBOSE": 1, "PRINT": 1}

structure = Hankel(m, m + r)

d_hat = grslra_online(x, structure, k, p, mu, r, params=grslra_params, scaling=scaling)