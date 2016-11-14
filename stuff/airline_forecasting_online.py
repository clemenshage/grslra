__author__ = 'cleha'

from grslra import grslra_online
import numpy as np
from grslra.structures import Hankel
from grslra.visualization import plot_lpnorm
from grslra.scaling import *
from grslra.tools import load_params

csvfile = np.genfromtxt('../data/airline_passengers_total_1996-2014.csv', delimiter=",", skip_header=4, skip_footer=3)
x = csvfile[0:-1,2]

p = 0.1
mu = 0.005

params_cg_U = {"t_init": 0.05}
grslra_params = {"VERBOSE": 1, "PRINT": 1, "II_slow": 3}
scaling = Scaling(centering=True)

N = x.size
k = 8
m = 24
n = N - m + 1
r = 6

hankel = Hankel(m, m + r)

x_hat = grslra_online(x, hankel, k, p, mu, r, params=grslra_params, params_cg_U=params_cg_U, scaling=scaling)