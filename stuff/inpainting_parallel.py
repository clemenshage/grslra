__author__ = 'cleha'
import sys

sys.path.append("..")

import numpy as np
from scipy import misc
from matplotlib import pyplot as plt
from grslra.structures import BlockHankel
from inpainting_patch import inpainting_patch
from grslra.tools import randomsampling
from joblib import Parallel, delayed
from grslra.visualization import plot_lpnorm
from grslra.scaling import *

PLOT = 0
VERBOSE = 0
PROFILE = 0

scaling = Normalization_Percentile(90, 0.5)

if PROFILE:
    pass

jobs = 4

rho = 0.1
(M, N) = (51, 51)
overlap = 26

k = 50
cg_params_U = {"rho": 0.5, "c": 1E-6, "thresh": 1E-6, "I": 10, "t_init": 1e-3, "reset_rate": 10, "t_min": 1E-8}
cg_params_Y = {"rho": 0.5, "c": 1E-4, "thresh": 1E-4, "I": 10, "t_init": 0.1, "reset_rate": 10, "t_min": 1E-8}
grslra_params = {"p": 2, "mu_start": 1e-8, "mu_end": 1e-8, "rho_0": 1E-3, "rho_I": 1, "I": 200, "cg_params_Y": cg_params_Y, "cg_params_U": cg_params_U, "PLOT": PLOT, "VERBOSE": VERBOSE, "PRINT": False, "scaling": scaling}


plot_lpnorm(grslra_params["p"], grslra_params["mu_start"], grslra_params["mu_end"])


m = (M + 1) / 2
n = m
a = (N + 1) / 2
b = a

blockhankel = BlockHankel(m, n, a, b)

im = misc.lena()
imheight = im.shape[0]
imwidth = im.shape[1]

A_full = randomsampling((imheight, imwidth), rho)

A_full, A_full_not = randomsampling((imheight, imwidth), rho)

im_sampled = A_full * im
im_sampled += A_full_not * 255 * np.random.rand(imheight, imwidth)

im_reconstructed = np.zeros(im.shape)

for ypos in xrange(0, imheight - N, N - overlap):
#    for xpos in xrange(0, imwidth - M, M - overlap):
    print ypos

        # plt.figure(10)
        # plt.ion()
        # plt.subplot(1, 3, 1)
        # plt.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='nearest')
        # plt.show()
        #
        # plt.figure(10)
        # plt.subplot(1, 3, 2)
        # plt.imshow(A * patch, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='nearest')
        # plt.show()

    patches = Parallel(n_jobs=jobs)(delayed(inpainting_patch)(im_sampled[ypos:ypos + N, xpos:xpos + M], blockhankel, k, A_full[ypos:ypos + N, xpos:xpos + M], grslra_params) for xpos in xrange(0, imwidth - M, M - overlap))
    xpositions = range(0, imwidth - M, M - overlap)

    for i in xrange(len(xpositions)):
        im_reconstructed[ypos+1:ypos + N, xpositions[i]+1:xpositions[i] + M] += 0.25 * patches[i][1:, 1:]

        # plt.figure(10)
        # plt.subplot(1, 3, 3)
        # plt.imshow(patch_hat, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='nearest')
        # plt.show()

    plt.figure(11, figsize=(15, 15))
    plt.ion()
    plt.imshow(im_reconstructed, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='nearest')
    plt.draw()
    plt.show()
    plt.savefig("lenna_reconstructed.png")



# if PROFILE:
#     profile = cProfile.Profile()
#     profile.enable()



# if PROFILE:
#     profile.disable()
#     profile.print_stats()


