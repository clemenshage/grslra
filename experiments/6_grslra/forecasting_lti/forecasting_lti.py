from grslra import testdata
from grslra.grslra_online import grslra_online
from grslra.structures import Hankel
from grslra.scaling import Scaling
import numpy as np

# The goal of this experiment is to identify an LTI system and to predict its impulse response in an online manner

PROFILE = 0
if PROFILE:
    import cProfile

N = 100
m = 20
k = 5
N_f = 3

_, x_0, U, Y = testdata.testdata_lti_outliers(N + N_f, m, k, rho=0.0)

scaling = Scaling(centering=True)
p = 0.1
mu= 1e-4

hankel = Hankel(m, m + N_f)

params_cg_U = {"t_init": 0.01}
grslra_params = {"VERBOSE": 1, "PRINT": 1}

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

x = x_0[:N]
l_grslra, angles, t = grslra_online(x, hankel, k, p, mu, N_f, params=grslra_params, params_cg_U=params_cg_U, scaling=scaling)

if PROFILE:
    profile.disable()
    profile.dump_stats("grslra.bin")

np.savez('result_forecasting_lti.npz', l_grslra=l_grslra, angles=angles, x=x, m=m, N=N, N_f=N_f)