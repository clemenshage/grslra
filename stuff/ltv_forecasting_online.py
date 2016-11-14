__author__ = 'cleha'

from grslra.structures import Hankel
from grslra import testdata, grslra_online
from grslra.visualization import plot_lpnorm
import matplotlib
from grslra.scaling import *
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})

scaling = Scaling_Percentile(100, 1.0)

N = 300
m = 20
k = 5
n = N - m + 1
r = 5

sigma = 0

d, _, _, _ = testdata.testdata_ltv_outliers(N, m, k, rho=0.1, amplitude=2, t=0.002, sigma=sigma)

p = 0.5

mu_end = np.maximum(9*(1-p) * sigma ** 2, 1e-4)

params_cg_U = {"t_init": 0.1, "rho_0": 0.9}
params_cg_Y = {"t_init": 10, "rho_0": 0.9}
grslra_params = {"VERBOSE": 1, "PRINT": 1}

#plot_lpnorm(p, mu_start, mu_end)

hankel = Hankel(m, m + r)

d_hat = grslra_online(d, hankel, k, p, mu_end, r, params=grslra_params, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y, scaling=scaling)