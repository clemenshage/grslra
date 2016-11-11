from grslra import testdata
from grslra.grslra_online import grslra_online
from grslra.structures import Hankel
from grslra.scaling import Scaling
import numpy as np

# The goal of this experiment is to predict the impulse response of an LTV in an online manner by fitting LTI systems to the observations

PROFILE = 0

if PROFILE:
    import cProfile

N = 200
m = 20
k = 5
N_f = 3

scaling = Scaling(centering=True)

p = 0.1
mu= 1e-4

x, x_0 = testdata.testdata_ltv_outliers(N + N_f, k, rho=0.0)

n = N + N_f - m + 1

hankel = Hankel(m, m + N_f)

params_cg_U = {"t_init": 0.01}
grslra_params = {"VERBOSE": 1, "PRINT": 1}

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

x = x_0[:N]
l_grslra, angles, t = grslra_online(x, hankel, k, p, mu, N_f, params=grslra_params, params_cg_U=params_cg_U, scaling=scaling, Omega=None)
# l_grslra = grslra_online(x_Omega, hankel, k, p, mu, N_f, params=grslra_params, params_cg_U=params_cg_U, scaling=scaling, Omega=Omega, N=N)

if PROFILE:
    profile.disable()
    profile.dump_stats("grslra.bin")

np.savez('result_forecasting_ltv.npz', l_grslra=l_grslra, angles=angles, x=x, m=m, N=N, N_f=N_f)