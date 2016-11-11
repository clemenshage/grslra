from matplotlib import pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})


data = np.load('errs_all.npz')
errs_rel = data["errs_rel"]
mulabels = data["mulabels"]

plt.figure(figsize=(15,5))
plt.hold(True)
colors=['k', 'b', 'g', 'r', 'm']

mus = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
for i in xrange(mus.__len__()):
    plt.semilogy(errs_rel[i], label="$\mu_\\textrm{end} = " + mulabels[i] + "$", linewidth=3, color=colors[i])

axes = plt.gca()
axes.set_ylim([0.004,1])
axes.set_xlim([0,95])
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()

plt.ylabel('$e_\\textrm{rel}$')
plt.legend()
plt.savefig('mu_eval.pdf', dpi=200)