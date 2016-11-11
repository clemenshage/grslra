from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

# this script is used to export selected foreground and background frames out of stored video sequences

framenumbers = [1806, 1813, 1820]

algos = ['ialm', 'godec', 'lmafit', 'k4_p0.1', 'k4_p0.4', 'k4_p0.7', 'k4_p1.0']
extensions = ['.mat', '.mat','.mat', '.npz', '.npz', '.npz', '.npz', '.npz']

for i in xrange(algos.__len__()):
    algoname = algos[i]
    ext = extensions[i]
    if ext == '.mat':
        data = loadmat(algoname + '/result_' + algoname + ext)
        dimensions = data["dimensions"][0]
    elif ext == '.npz':
        data = np.load('result_' + algoname + ext)
        dimensions = data["dimensions"]
    else:
        exit()

    L = data["L"]
    X_masked = data["X_masked"]

    for framenumber in framenumbers:
        fg_imagename= 'frames/' + algoname.replace('.', '_') + '_fg_{:03d}'.format(framenumber) + '.png'
        bg_imagename= 'frames/' + algoname.replace('.', '_') + '_bg_{:03d}'.format(framenumber) + '.png'

        frame = np.reshape(L[:,framenumber+1], dimensions, order='F')
        plt.figure()
        fig=plt.imshow(frame, cmap="gray", clim=(0, 255))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(bg_imagename, dpi=300, bbox_inches='tight')
        plt.close()

        frame = np.reshape(X_masked[:,framenumber+1], dimensions, order='F')
        plt.figure()
        fig=plt.imshow(frame, cmap="gray", clim=(0, 255))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(fg_imagename, dpi=300, bbox_inches='tight')
        plt.close()