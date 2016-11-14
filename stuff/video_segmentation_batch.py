import numpy as np
from grslra.visualization import plot_lpnorm
from grslra.scaling import *
from grslra.video import Video
from grslra.grpca import grpca
import time


OFFLINE_MEDIANSUB = 1
PROFILE = 0

rate_Omega = 1

if PROFILE:
    import cProfile

video_in = Video()
video_in.create_from_image_sequence('/home/cleha/data/video_sequences/escalator/gray/*.bmp')
#video_in.save_to_file('movie_small')
#video_in.load_from_file('movie_small')
video_in.save_to_npz('movie_huge')
#video_in.load_from_file('movie_big')
#video_in.play()

X = video_in.X
dimensions=X.shape

if OFFLINE_MEDIANSUB:
    median = np.atleast_2d(np.median(X, axis=1)).T
    X -= median

if rate_Omega == 1:
    X_Omega = X
    Omega = None
else:
    card_Omega = np.int(np.round(rate_Omega * X.shape[0] * X.shape[1]))
    Omega = (np.random.choice(X.shape[0], card_Omega, replace=True), np.random.choice(X.shape[1], card_Omega, replace=True))
    X_Omega = X[Omega]

if OFFLINE_MEDIANSUB:
    scaling = Scaling(percentile=100, val_at_percentile=1.0)
else:
    scaling = Scaling(centering=True, percentile=100, val_at_percentile=1.0)

p = 0.1

tau = 20.0
mu_end = (tau / 255.0) / 3


params_grpca = {"PRINT": None, "VERBOSE": 0, "SMMPROD": True}

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

pvalues=[0.1, 0.4, 0.7, 1.0]

for k in xrange(4,10):
    print "k = ", k
    for j in xrange(4):
        p = pvalues[j]
        print "p = ", p

        t_start = time.time()
        U, Y = grpca(X_Omega, k, p , mu_end, params=params_grpca, Omega=Omega, dimensions=dimensions, scaling=scaling)

        t = time.time() - t_start
        print "Finished in ", t, " seconds"

        if PROFILE:
            profile.disable()
            profile.dump_stats("profile.bin")

        L = np.dot(U,Y)

        if OFFLINE_MEDIANSUB:
            L += median
            X_in = X + median
        else:
            X_in = X


        L = np.maximum(L, 0.0)
        L = np.minimum(L, 255.0)

        video_out = Video()
        video_out.create_from_matrix(L, (video_in.rows, video_in.cols))
        #video_out.save_to_file('lowrank')
        video_out.write_video('lowrank_' + '{:d}'.format(k) + "_" + '{:1.1f}'.format(p), scaling=1)

        mask = np.array(np.abs(X_in - L) > tau, dtype=np.int8)

        video_out.create_from_matrix(mask, (video_in.rows, video_in.cols))
        from scipy.signal import medfilt2d as mfilter
        for i in xrange(video_out.n):
            video_out.frames[:,:,i] = mfilter(video_out.frames[:,:,i], kernel_size=3)

        mask_reshaped = np.reshape(video_out.frames,(video_out.m, video_out.n), order='F')

        X_masked = X_in * mask_reshaped
        X_masked[mask_reshaped==0] = 255.0

        video_out.create_from_matrix(X_masked, (video_in.rows, video_in.cols))
        #video_out.save_to_file('sparse')
        video_out.write_video('sparse_' + '{:d}'.format(k) + "_" + '{:1.1f}'.format(p), scaling=1)
        #video_out.play()


