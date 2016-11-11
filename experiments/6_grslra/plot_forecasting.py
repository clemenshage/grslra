from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})

# This function visualizes the results of the LTI/LTV forecasting experiments

name ='forecasting_lti'

data = np.load(name + '/' + 'result_' + name + '.npz')

plt.figure(figsize=(15,5))
plt.hold(True)

plt.plot(np.hstack((0, data["x"])), label="ground truth", linewidth=3, color='grey',zorder=0)
plt.plot(np.hstack((0, data["l_grslra"])), label="predicted", linewidth=3, color='g', zorder=1)

plt.xlim(2 * data["m"] - 1, data["N"] + data["N_f"])

axes = plt.gca()
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()

axes.legend()#loc="upper right"
plt.savefig(name + '/' + name + '.pdf', dpi=200)



plt.figure(figsize=(15,5))
plt.hold(True)

plt.plot(np.hstack((0, data["angles"])), linewidth=3, color='black',zorder=0)

plt.xlim(2 * data["m"] - 1, data["N"])
plt.ylim(0, 90)
plt.ylabel("subspace angle")

axes = plt.gca()
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()

plt.savefig(name + '/' + name + '_angles.pdf', dpi=200)



