from matplotlib import pyplot as plt
import numpy as np
import matplotlib


matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})
colors = ['k', 'b', 'g', 'r', 'm']


data = np.load('result_subsampling.npz')
times = data["times"]
dims = data["dims"]
paramfiles = data["paramfiles"]
nofparamfiles = paramfiles.__len__()
plt.figure(figsize=(15, 5))
plt.hold(True)

for j in xrange(nofparamfiles):
    plt.loglog(dims**2, times[:, j], linewidth=3, color=colors[j], label="$\\textrm{" + paramfiles[j] + "}$")

axes = plt.gca()
axes.set_ylim([1,1e4])
axes.set_xlim([dims[0]**2,dims[-1]**2])
plt.grid(b=True, which='both', color='0.65', linestyle='-')
plt.tight_layout()

plt.ylabel('time (s)')
plt.xlabel('$m^2$')
plt.legend()
plt.savefig('subsampling_eval.pdf', dpi=200)
plt.close()


data = np.load('result_smmprod.npz')
times = data["times"]
dims = data["dims"]
paramfiles = data["paramfiles"]
nofparamfiles = paramfiles.__len__()
plt.figure(figsize=(15, 5))
plt.hold(True)

for j in xrange(nofparamfiles):
    plt.loglog(dims[:10]**2, times[:10, j], linewidth=3, color=colors[j], label="$\\textrm{" + paramfiles[j] + "}$")

axes = plt.gca()
axes.set_ylim([1,1e4])
axes.set_xlim([dims[0]**2,dims[9]**2])
plt.grid(b=True, which='both', color='0.65', linestyle='-')
plt.tight_layout()

plt.ylabel('time (s)')
plt.xlabel('$m^2$')
plt.legend()
plt.savefig('smmprod_eval.pdf', dpi=200)
plt.close()