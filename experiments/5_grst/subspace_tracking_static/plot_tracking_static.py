from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})

pvalues = [0.1, 0.4, 0.7, 1.0]

plt.figure(figsize=(15,5))
plt.hold(True)
colors=['k', 'b', 'g', 'r', 'm']

k=10
for i in xrange(pvalues.__len__()):
    p = pvalues[i]
    data = np.load('errs' + '_k_{:d}'.format(k) + '_p_{:1.1f}'.format(p) + '.npz')
    errs_rel_av = data["errs_rel_av"]
    avstep = data["avstep"]
    n = errs_rel_av.shape[0] * avstep
    plt.plot(np.arange(0, n, avstep), errs_rel_av, label="$p = {:1.1f}$".format(pvalues[i]), linewidth=3, color=colors[i])

axes = plt.gca()
axes.set_ylim([0,1.6])
axes.set_xlim([0,n])
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()

plt.ylabel('$e_\\textrm{rel}$')
plt.xlabel('samples')
plt.legend(loc="right")
plt.savefig('tracking_static.pdf'.format(k), dpi=200)
plt.close()

plt.figure(figsize=(15,5))
plt.hold(True)

for i in xrange(pvalues.__len__()):
    p = pvalues[i]
    data = np.load('errs' + '_k_{:d}'.format(k) + '_p_{:1.1f}'.format(p) + '.npz')
    angles_av = data["angles_av"]
    n = angles_av.shape[0] * avstep
    plt.plot(np.arange(0, n, avstep), angles_av, label="$p = {:1.1f}$".format(pvalues[i]), linewidth=3, color=colors[i])

axes = plt.gca()
axes.set_ylim([0,90])
axes.set_xlim([0,n])
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()

plt.ylabel('$e_\\textrm{sub}$')
plt.xlabel('samples')
plt.legend(loc='right')
plt.savefig('tracking_static_angles.pdf', dpi=200)
plt.close()
