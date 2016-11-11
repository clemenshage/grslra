from os import chdir
from grslra.visualization import phaseplot
import numpy as np
from scipy.io import loadmat

# This function can be used to visualize the phaseplot experiments of the MATLAB algorithms

algoname = 'rmc'

folder='phasetransitions/' + algoname
algoname = 'rmc_fast'
resultfile= 'result_' + algoname

chdir(folder)
data = loadmat(resultfile + '.mat')
errs = data['result'][:,:,0]
maxval = data['maxval']
step = data['step']
values = np.int(np.round(maxval / step))

phaseplot(np.log10(errs), (-8, -1), values, maxval, 'pt_' + algoname + '.pdf')