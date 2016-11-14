import numpy as np
from scipy.io import loadmat
from grslra.visualization import plot_lpnorm
from grslra.scaling import Scaling
from grslra.grpca import grpca
import smmprod

PROFILE = 1
PLOT = None
VERBOSE = 1
PRINT = None


rate = 0.1
k = 5

data = loadmat('../data/movielens100k.mat')
X_in = data["X"]
Omega = np.where(X_in)
card_Omega = Omega[0].size

ix = np.random.choice(card_Omega, np.int(np.round(rate * card_Omega)), replace=False)

tmp = np.zeros((card_Omega,))
tmp[ix] = 1

ix_train = np.where(tmp == 0)
ix_test = np.where(tmp == 1)

Omega_train = Omega[0][ix_train], Omega[1][ix_train]
Omega_test = Omega[0][ix_test], Omega[1][ix_test]

X_train = X_in[Omega_train]
X_test = X_in[Omega_test]

scaling = Scaling(100, 1.0)

p = 0.5
mu_start = 1E-1
mu_end = 1E-2
#p=2.0
#mu_start=1E-7
#mu_end=1E-8

#params_cg_U = {"rho_0": 0.5, "I": 10, "t_init": 1e5}
#params_cg_Y = {"rho_0": 0.5, "I": 20}

grpca_params = {"mu_start": mu_start, "min_progress":1E-3, "SMMPROD":True, "samplesize": None, "VERBOSE":1, "PRINT": None}
grpca_params["PCA_INIT"] = True

plot_lpnorm(p, mu_start, mu_end)

if PROFILE:
    import cProfile
    profile = cProfile.Profile()
    profile.enable()

U, Y = grpca(X_train, k, p, mu_end, params=grpca_params, Omega=Omega_train, dimensions=X_in.shape, kappa=None)#, params_cg_U=params_cg_U, params_cg_Y=params_cg_Y )

if PROFILE:
    profile.disable()
    profile.dump_stats("profile.bin")

UY_predicted = smmprod.smmprod_c(U, Y, Omega_test)

err_stupid = np.sqrt(np.mean((X_test - np.mean(X_train.flatten())) ** 2))

err_RMSE = np.sqrt(np.mean((X_test - UY_predicted) ** 2))

print "stupid guess: ", err_stupid
print "prediction: ", err_RMSE


