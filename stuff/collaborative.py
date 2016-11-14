__author__ = 'cleha'
import numpy as np
from grslra.structures import SkewSymmetric
from grslra.grslra_batch import grslra_batch
from grslra.visualization import plot_lpnorm

PROFILE = 1

if PROFILE:
    import cProfile

# set the parameters of the toy data
users = 1000
items = 500
levels = 5
error_rate = 0.1
sampling_rate = 0.1

l = np.random.choice(levels, items, replace=True)
l = np.sort(l)

L_0 = np.sign(np.outer(l, np.ones(items, )) - np.outer(np.ones(items, ), l))

X = np.zeros((users, items, items))
Omega = np.zeros_like(X)

for i in xrange(users):
    # initialize empty observation
    X_i = np.zeros((items, items))

    # choose which items will be observed and get their indices (only upper triangle)
    Omega_i = np.triu(1.0 * (np.random.rand(items, items) < 2*sampling_rate))
    np.fill_diagonal(Omega_i, 0)
    Omega_ix = np.where(Omega_i)
    # copy observations from L_0
    X_i[Omega_ix] = L_0[Omega_ix]

    # pick indices that will suffer from errors
    if Omega_ix[0].size > 0:
        tmp = np.random.choice(Omega_ix[0].size, np.int(np.round(error_rate * Omega_ix[0].size)))
    else:
        tmp = []
    E_ix = Omega_ix[0][tmp], Omega_ix[1][tmp]

    entries = X_i[E_ix]
    err = np.random.choice([1, 2], entries.size, replace=True)
    # add error and perform modulo operation
    entries = np.mod(entries + err, 2)
    # write to matrix
    X_i[E_ix] = entries

    # add lower triangle
    X_i -= X_i.T

    Omega_i += Omega_i.T + np.eye(items)

    X[i, :, :] = X_i
    Omega[i, :, :] = Omega_i

    tmp = np.where(Omega_i)
    Omega_i_ix = (tmp[0],tmp[1])

    X_i_ix = X_i[tmp]

    if i:
        Xlist.append(X_i_ix)
        Omegalist.append(Omega_i_ix)
    else:
        Xlist = [X_i_ix,]
        Omegalist = [Omega_i_ix,]


structure = SkewSymmetric(items)

scaling = None

k = 2
p = 0.1
mu_end = 1e-4

grslra_params = {"VERBOSE": 3}
plot_lpnorm(p, 0.1, mu_end)

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

L_hat, _, _ = grslra_batch(Xlist, structure, k, p, mu_end, params=grslra_params, Omega=Omegalist, dimensions=(users, items, items))


if PROFILE:
    profile.disable()
    profile.dump_stats("profile.bin")

print "\nRelative reconstruction error ||L_0 - L||_F / ||L_0||_F = ", np.linalg.norm(L_0 - L_hat, 'fro') / np.linalg.norm(L_0, 'fro')


def modify_observations(observations):
    # shift values to be (0, 1, 2) so that we can use modulo
    observations += 1
    err = np.random.choice([1, 2], observations.size, replace=True)
    # add error and perform modulo operation
    observations = np.mod(observations + err, 2)
    # shift back
    observations -= 1
    return observations
