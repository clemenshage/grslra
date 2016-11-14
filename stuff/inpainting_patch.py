__author__ = 'cleha'

import numpy as np

from grslra import grslra_batch
from grslra.scaling import Scaling
from grslra.tools import subspace_angle


def inpainting_patch(patch, A, blockhankel, k, p, mu, grslra_params, U_init=None, Y_init=None, params_cg_U=None):
    scaling = Scaling()

    (M, N) = patch.shape
    x = np.double(patch.flatten('F'))
    A = A.flatten('F')
    Omega = np.where(A)[0]
    x_Omega = x[Omega]

    if Y_init is not None:
        Y_init=np.roll(Y_init,-blockhankel.m,axis=1)
        Y_init[:, -1 - blockhankel.m:] = 0

    l, U, Y = grslra_batch(x_Omega, blockhankel, k, p, mu, params=grslra_params, Omega=Omega, params_cg_U=params_cg_U, U_init=U_init, Y_init=Y_init)
    if U_init is not None:
        print "angle: ", subspace_angle(U, U_init)

    patch_hat = np.reshape(l, (M, N), order='F')

    return patch_hat, U, Y