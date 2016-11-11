import matplotlib
import numpy as np
from matplotlib import pyplot as plt

try:
    import cv2
    OPENCV = True
except ImportError:
    OPENCV = False
    "Error: Unable to import cv2. If you want to enable video output please install OpenCV first."

try:
    import pims
except ImportError:
    PIMS = False
    "Error: Unable to import pims. If you want to create a video object from an image sequence please install PIMS first."


class Video:
    # This class can be used to handle videos as matrices with stacked frames. It requires OpenCV for video export
    def __init__(self):
        self.frames = None
        self.X = None
        self.m = None
        self.n = None
        self.rows = None
        self.cols = None

    def create_from_image_sequence(self, folder):
        if not PIMS:
            print "Failed to create video object from image sequence. PIMS is not available."
        else:
            images = pims.ImageSequence(folder, as_grey=True)
            self.n = images.__len__()
            self.rows, self.cols = images[0].shape
            self.m = self.rows * self.cols
            self.frames = np.zeros((self.rows, self.cols, self.n))
            self.X = np.zeros((self.m, self.n))
            for i in xrange(self.n):
                print "reading input frame ", i
                self.frames[:, :, i] = images.get_frame(i)
                self.X[:, i] = np.reshape(images.get_frame(i), (self.m,), order='F')

    def create_from_matrix(self, X, dimensions):
        self.X = X
        self.m, self.n = X.shape
        self.rows, self.cols = dimensions
        self.frames = np.zeros((self.rows, self.cols, self.n))
        for i in xrange(self.n):
            self.frames[:, :, i] = np.reshape(self.X[:, i], (self.rows, self.cols), order='F')

    def play(self, first=None, last=None):
        matplotlib.use('TkAgg')
        fig = plt.figure()
        fig.add_subplot(111)

        def animate(firstframe, lastframe):
            data = self.frames[:, :, 0]
            im = plt.imshow(data, cmap='gray', interpolation='Nearest')
            if firstframe is None:
                firstframe = 0
            if lastframe is None:
                lastframe = self.n
            for i in xrange(firstframe, lastframe):
                data = self.frames[:, :, i]
                im.set_data(data)
                fig.canvas.draw()

        fig.canvas.manager.window.after(100, animate(first, last))
        plt.show()

    def save_to_npz(self, filename):
        np.savez(filename, X=self.X, frames=self.frames, dimensions=(self.rows, self.cols))

    def load_from_npz(self, filename):
        fname = filename + '.npz'
        video = np.load(fname)
        self.X = video["X"]
        self.rows, self.cols = video["dimensions"]
        self.m, self.n = self.X.shape
        self.frames = video["frames"]

    def write_video(self, name, scaling=4.0):
        if not OPENCV:
            print "Failed to write video. OpenCV is not available."
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            filename = name + '.mpg'
            video = cv2.VideoWriter(filename, fourcc, 25.0, (np.int(scaling * self.cols), np.int(scaling * self.rows)), isColor=0)
            for i in xrange(self.n):
                frame = self.frames[:, :, i]
                frame_scaled = cv2.resize(frame, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)
                video.write(np.uint8(frame_scaled))

            cv2.destroyAllWindows()
            video.release()
