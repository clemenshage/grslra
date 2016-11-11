from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})

pvalues = [0.1, 0.4, 0.7, 1.0]

plt.figure(1, figsize=(15,5))
plt.hold(True)
plt.figure(2, figsize=(15,5))
plt.hold(True)
colors=['k', 'b', 'g', 'r', 'm']

k = 5

for i in xrange(pvalues.__len__()):
    p = pvalues[i]
    data = np.load('errs' + '_k_{:d}'.format(k) + '_p_{:1.1f}'.format(p) + '.npz')
    avstep = data["avstep"]
    errs_rel_av = data["errs_rel_av"]
    angles_av = data["angles_av"]
    n = (errs_rel_av.shape[0]-1) * avstep
    plt.figure(1)
    plt.plot(np.arange(0, n + 1, avstep), errs_rel_av, label="$p = {:1.1f}$".format(pvalues[i]), linewidth=3, color=colors[i])
    plt.figure(2)
    plt.plot(np.arange(0, n + 1, avstep), angles_av, label="$p = {:1.1f}$".format(pvalues[i]), linewidth=3, color=colors[i])

plt.figure(1)
axes = plt.gca()
axes.set_ylim([0,1])
axes.set_xlim([0,n+1])
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()
plt.ylabel('$e_\\textrm{rel}$')
plt.xlabel('samples')
plt.legend()
plt.savefig('tracking_static_incomplete_err.pdf'.format(k), dpi=200)

plt.figure(2)
axes = plt.gca()
axes.set_ylim([0,90])
axes.set_xlim([0,n+1])
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()
plt.ylabel('$e_\\textrm{rel}$')
plt.xlabel('samples')
plt.legend()
plt.savefig('tracking_static_incomplete_angle.pdf'.format(k), dpi=200)


