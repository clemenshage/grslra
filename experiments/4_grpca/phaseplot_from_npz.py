from os import chdir
from grslra.visualization import phaseplot
import numpy as np

# This function can be used to visualize the phaseplot experiments of the Python algorithms

folder='pvalues/pvalues_ialm'
resultfile= 'result_1.0'

chdir(folder)
data = np.load(resultfile + '.npz')
errs = data['result'][:,:,0]
maxval = data['maxval']
step = data['step']
values = np.int(np.round(maxval / step))

phaseplot(np.log10(errs), (-3, -1), values, maxval, "pt_ialmdata_1_0.pdf")