from scipy.io import loadmat
from grslra.video import Video

# This script reads the decomposition results of the algorithms implemented in MATLAB and outputs video sequences

for algoname in ['ialm', 'godec', 'lmafit']:

    bg_video = Video()
    fg_video = Video()
    videoscaling = 2

    filename = 'result_' + algoname + '.mat'

    data = loadmat(algoname + '/result_' + algoname + '.mat')
    dimensions = data["dimensions"][0]
    L = data["L"]
    X_masked = data["X_masked"]

    bg_video.create_from_matrix(L, dimensions)
    fg_video.create_from_matrix(X_masked, dimensions)

    bg_video.write_video('background_' + algoname, scaling=videoscaling)
    fg_video.write_video('foreground_' + algoname, scaling=videoscaling)