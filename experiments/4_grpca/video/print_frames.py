from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

# This script prints selected frames of the stored escalator video sequence

data = loadmat('escalator_130p.mat')

X = data["X"]
dimensions = data["dimensions"][0]

framenumbers = [1806, 1813, 1820]

for framenumber in framenumbers:
    imagename = 'frames/escalator_' + '{:03d}'.format(framenumber) + '.png'

    frame = np.reshape(X[:,framenumber+1], dimensions, order='F')
    plt.figure()
    fig=plt.imshow(frame, cmap="gray", clim=(0, 255))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(imagename, dpi=300, bbox_inches='tight')
    plt.close()