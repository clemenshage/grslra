import numpy as np
from scipy.io import loadmat
from grslra.structures import SkewSymmetric
from grslra.grslra_batch import grslra_collaborative_batch
from grslra.visualization import plot_lpnorm

PROFILE = 1
PLOT = None
VERBOSE = 2
PRINT = None

data = loadmat('../data/movielens100k.mat')
X_in = data["X"]
X = None
Omega = None

users, items = X_in.shape

for i in xrange(users):
    print i

    x = X_in[i,:]

    Omega_i = np.where(np.outer(x,x))

    Z = np.sign(np.outer(x, np.ones(items,)) - np.outer(np.ones(items,), x))

    X_i = Z[Omega_i]

    if i:
        X.append(X_i)
        Omega.append(Omega_i)
    else:
        X = [X_i,]
        Omega = [Omega_i,]

structure = SkewSymmetric(items)

scaling = None

p = 0.1
mu_start = 0.1
mu_end = 1e-4


cg_params_U = {"rho": 0.9, "c": 1E-6, "thresh": 1E-6, "I": 100, "t_init": 1e4, "reset_rate": 10, "t_min": 1E-8}
cg_params_Y = {"rho": 0.9, "c": 1E-4, "thresh": 1E-4, "I": 100, "t_init": 1e8, "reset_rate": 10, "t_min": 1E-8}

grslra_params = {"p": p, "min_progress": 0.0001, "mu_start": mu_start, "mu_end": mu_end, "rho_start": 1, "rho_end": 1e6, "cg_params_Y": cg_params_Y, "cg_params_U": cg_params_U, "PLOT": PLOT, "VERBOSE": VERBOSE, "PRINT": PRINT, "scaling": scaling}
plot_lpnorm(grslra_params["p"], grslra_params["mu_start"], grslra_params["mu_end"])

if PROFILE:
    import cProfile
    profile = cProfile.Profile()
    profile.enable()

L_hat, _, _, _ = grslra_collaborative_batch(X, structure, 4, Omega, grslra_params)

if PROFILE:
    profile.disable()
    profile.dump_stats("profile_big.bin")

np.save('L_hat.npy','L_hat')

tmp=0