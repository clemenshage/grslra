import numpy as np
from grslra.video import Video

# This script reads the decomposition results of the algorithms implemented in Python and outputs video sequences

for algoname in ['k4_p0.1', 'k4_p0.4', 'k4_p0.7', 'k4_p1.0']:

    bg_video = Video()
    fg_video = Video()
    videoscaling = 2

    data = np.load('result_' + algoname + '.npz')
    dimensions = data["dimensions"]
    L = data["L"]
    X_masked = data["X_masked"]

    bg_video.create_from_matrix(L, dimensions)
    fg_video.create_from_matrix(X_masked, dimensions)

    bg_video.write_video('background_' + algoname.replace('.', '_'), scaling=videoscaling)
    fg_video.write_video('foreground_' + algoname.replace('.', '_'), scaling=videoscaling)