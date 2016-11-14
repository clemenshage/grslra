__author__ = 'cleha'

import numpy as np
from scipy import misc
from matplotlib import pyplot as plt

from grslra.structures import BlockHankel
from inpainting_patch import inpainting_patch
from grslra.tools import randomsampling
from grslra.scaling import *
import time

PROFILE = 1

if PROFILE:
    import cProfile

density = 0.2

p=2.0
mu=1e-2

scaling = Scaling()
(M, N) = (19,19)
overlap = 18

k = 10


m = (M + 1) / 2
n = m
a = (N + 1) / 2
b = a

grslra_params={"VERBOSE": 1}
params_cg_U = {"t_init": 5e-5}

blockhankel = BlockHankel(m, n, a, b, memsave=M > 60)

im = misc.imread('../data/lena.jpg', flatten=True)[200:400, 200:400]
imheight = im.shape[0]
imwidth = im.shape[1]

A_full, A_full_not = randomsampling((imheight, imwidth), density)

im_sampled = A_full * im
# im_sampled += A_full_not * 255 * np.random.rand(imheight, imwidth)

im_reconstructed = np.zeros(im.shape)

U = None
Y = None

printit=0
for ypos in xrange(0, imheight - N, N - overlap):
    for xpos in xrange(0, imwidth - M, M - overlap):
        patch_orig = im[ypos:ypos + N, xpos:xpos + M]
        patch = im_sampled[ypos:ypos + N, xpos:xpos + M]
        A = A_full[ypos:ypos + N, xpos:xpos + M]

        plt.figure(10, figsize=(20,15))
        plt.ion()
        plt.subplot(2, 3, 1)
        plt.imshow(patch_orig, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='nearest')

        plt.figure(10)
        plt.subplot(2, 3, 2)
        plt.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='nearest')
        plt.draw()

        if PROFILE:
            profile = cProfile.Profile()
            profile.enable()

        patch_hat, U, Y = inpainting_patch(patch, A, blockhankel, k, p, mu, grslra_params, U_init=U, params_cg_U=params_cg_U, Y_init=Y)

        if PROFILE:
            profile.disable()
            profile.dump_stats("profile.bin")

        im_reconstructed[ypos:ypos + N, xpos:xpos + M] += patch_hat
        im_reconstructed[ypos + int((N-1)/2), xpos + int((M-1)/2)] = patch_hat[int((N-1)/2), int((M-1)/2)]

        plt.figure(10)
        plt.subplot(2, 3, 3)
        plt.imshow(patch_hat, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='nearest')
        plt.subplot(2,3,4)
        plt.imshow(blockhankel.struct_from_vec(patch_hat.flatten('F')), interpolation='nearest',  cmap=plt.cm.gray, vmin=0, vmax=255)
        plt.subplot(2,3,5)
        plt.imshow(U, interpolation='nearest',  cmap=plt.cm.gray)
        plt.subplot(2,3,6)
        plt.imshow(Y, interpolation='nearest',  cmap=plt.cm.gray)
        plt.draw()
        printit+=1
        plt.savefig('/tmp/'+ '{:03d}'.format(printit) + '.png')

    plt.figure(11, figsize=(15, 15))
    plt.ioff()
    plt.imshow(im_reconstructed, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='nearest')
    plt.draw()
    plt.savefig("/tmp/lenna_reconstructed.png")










