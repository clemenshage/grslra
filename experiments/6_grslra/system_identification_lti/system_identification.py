from grslra import testdata
from grslra.grslra_batch import grslra_batch, slra_by_factorization
from grslra.structures import Hankel
from grslra.scaling import Scaling
import numpy as np
import time

# The goal of this experiment is to identify an LTI system from a noisy outlier-contaminated and subsampled observation of its impulse response

PROFILE = 0

if PROFILE:
    import cProfile

N = 80
m = 20
k = 5

sigma=0.05
outlier_rate = 0.05
outlier_amplitude = 1

rate_Omega=0.5
N_f = 20

scaling = Scaling(centering=True)

p = 0.1

x, x_0, U, Y = testdata.testdata_lti_outliers(N + N_f, m, k, rho=outlier_rate, amplitude=outlier_amplitude, sigma=sigma)

# determine scaling factor
scaling.scale_reference(x)
mu = (1-p) * (3 * sigma / scaling.factor) ** 2

# draw sampling set
card_Omega = np.int(np.round(rate_Omega * N))
Omega = np.random.choice(N, card_Omega, replace=False)
# create binary support vectors for Omega and Omega_not
entries = np.zeros((N + N_f, ))
entries[Omega] = 1
entries_not = np.ones_like(entries) - entries
# set unobserved entries in x to zero
x *= entries
x_Omega = x[Omega]

n = N + N_f - m + 1

hankel = Hankel(m, n)

grslra_params = {"PRINT": None, "VERBOSE": 1}

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

t_start = time.time()
l_grslra, U, Y = grslra_batch(x_Omega, hankel, k, p, mu, params=grslra_params, Omega=Omega, x_0=x_0, scaling=scaling)
t_grslra = time.time() - t_start

if PROFILE:
    profile.disable()
    profile.dump_stats("grslra.bin")

print "error GRSLRA: ", np.linalg.norm(l_grslra - x_0) / np.linalg.norm(x_0)
print "time GRSLRA: ", t_grslra

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

t_start = time.time()
l_slrabyF = slra_by_factorization(x_Omega, m, k, PRINT=0, x_0=x_0, Omega=Omega, N=N + N_f)
t_slrabyf = time.time() - t_start

if PROFILE:
    profile.disable()
    profile.dump_stats("slrabyf.bin")

print "error SLRA by F: ", np.linalg.norm(l_slrabyF - x_0) / np.linalg.norm(x_0)
print "time SLRA by F: ", t_slrabyf

np.savez('result_sysid_lti.npz', x_Omega=x_Omega, Omega=Omega, x_0=x_0, t_grslra=t_grslra, l_grslra=l_grslra, t_slrabyf=t_slrabyf, l_slrabyF=l_slrabyF)