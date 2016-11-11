from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})

# This function visualizes the results of the batch system identification experiment

data = np.load('result_sysid_lti.npz')

plt.figure(figsize=(15,5))
plt.hold(True)

plt.plot(data["x_0"], label="Ground Truth", linewidth=3, color='grey',zorder=0)
plt.scatter(data["Omega"], data["x_Omega"], label="Observations", color='k',zorder=3)
plt.plot(data["l_grslra"], label="GRSLRA", linewidth=3, color='g', zorder=1)
plt.plot(data["l_slrabyF"], label="SLRAbyF", linewidth=3, color='r', zorder=2)

axes = plt.gca()
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()

handles, labels = axes.get_legend_handles_labels()

order = [0, 3, 2, 1]

leg=axes.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, frameon=True)#loc="upper right"
leg.get_frame().set_edgecolor('k')
plt.savefig('sysid_lti.pdf', dpi=200)