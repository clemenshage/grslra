import numpy as np
from grslra.video import Video
from grslra.scaling import *

scaling = Normalization_Percentile(33, 0.1)

video = Video()

video.load_from_npz('sparse')
video.write_video('videos/sparse')
S = video.X
dimensions=(video.rows, video.cols)
mask = np.abs(S) > 5.0

video.load_from_npz('movie_big')
X_masked = video.X * mask

video.create_from_matrix(X_masked,dimensions=dimensions)
video.write_video('test.mpg')
# video.play(last=1000)
#
#
# video.load_from_file('movie')
#
# X = video.X
# dimensions=(video.rows, video.cols)
# X_scaled = scaling.scale_reference(X)
#
# video_scaled = Video()
# video_scaled.create_from_matrix(X_scaled, dimensions)
#
# video.play(last=100)
# video_scaled.play(last=100)
#
#
