import numpy as np
from scipy.io import loadmat
from grslra.scaling import Scaling
from grslra.video import Video
from grslra.grpca import grpca
import time
from scipy.signal import medfilt2d as mfilter
from sys import argv

# This experiment performs video segmentation on the escalator sequence from Li et al (2004)
# The cost function parameter p must be provided as an inline parameter

PROFILE = 0
if PROFILE:
    import cProfile

# Set to a value < 1.0  if the video should be subsampled
rate_Omega = 1.0
# It is faster to estimate and subtract the median from the fully observed video
OFFLINE_MEDIANSUB = 1

# size of the median filter kernel
kernel_size = 3

# Threshold in intensity levels
tau = 20.0

video_in = Video()
data=loadmat('escalator_130p.mat')
video_in.create_from_matrix(data["X"].astype(np.double), data["dimensions"][0])

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

# Don't actually scale yet, only determine the scaling factor for computing mu
scaling.scale_reference(X_Omega)

p = float(argv[1])
mu_opt = (1 - p) * (tau / scaling.factor) ** 2
# for p==1, mu_opt would be 0. Instead use the mu value for p=0.9
mu_end = np.maximum(mu_opt, 0.1 * (tau/ scaling.factor) ** 2)
print "p = ", p
print "mu_end = ", mu_end

params_grpca = {"PRINT": None, "VERBOSE": 1, "c_mu": 0.5}

if PROFILE:
    profile = cProfile.Profile()
    profile.enable()

k = 4
print "k = ", k

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

video_dummy = Video()

filename = 'result_' + 'k{:d}'.format(k) + "_" + 'p{:1.1f}'.format(p)
if rate_Omega < 1:
    filename += '_{:d}perc'.format(int(np.round(100 * rate_Omega)))

# create a segmentation mask and apply a median filter to it
mask = np.array(np.abs(X_in - L) > tau, dtype=np.int8)
video_dummy.create_from_matrix(mask, (video_in.rows, video_in.cols))
for i in xrange(video_dummy.n):
    video_dummy.frames[:,:,i] = mfilter(video_dummy.frames[:,:,i], kernel_size=kernel_size)
mask_reshaped = np.reshape(video_dummy.frames,(video_dummy.m, video_dummy.n), order='F')

# extract the foreground elements using this mask
X_masked = X_in * mask_reshaped
X_masked[mask_reshaped==0] = 255.0

np.savez(filename, X_masked=X_masked.astype(np.uint8), L=L.astype(np.uint8), dimensions=(video_dummy.rows, video_dummy.cols), t=t)