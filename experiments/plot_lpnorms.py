import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.rcParams.update({'font.size': 24})
matplotlib.rcParams.update({'text.usetex': True})


def lpnorm_scaled(x, p, mu):
    return (lpnorm(x, p, mu) - lpnorm(0, p, mu)) / (lpnorm(1, p, mu) - lpnorm(0, p, mu))


def lpnorm(x, p, mu):
    return (mu + x * x) ** (p / 2.0)

pvalues=[2.0, 1.0, 0.7, 0.4, 0.1]
mu = 1e-12
colors=['k', 'b', 'g', 'r', 'm']

x = np.linspace(-1, 1, 1001)

plt.figure(figsize=(15,8))


for i in xrange(pvalues.__len__()):
    p = pvalues[i]
    plt.plot(x, lpnorm_scaled(x, p, mu), color=colors[i], label='$p={:1.1f}$'.format(pvalues[i]), linewidth=3)

plt.legend()
axes = plt.gca()
axes.set_ylim([0,1])
axes.set_xlim([-1,1])
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()

plt.legend()
plt.savefig('lpnorm_fixedmu.pdf', dpi=200)


muvalues=[0.01, 1e-3, 1e-4]
labels = ["$\\ell_2$", "$\\ell_1$", "$\\mu=0.01$", "$\\mu=0.001$", "$\\mu=10^{-4}$"]


plt.figure(figsize=(15,8))
plt.plot(x, lpnorm_scaled(x, 2.0, mu), color=colors[0], label=labels[0], linewidth=3)
plt.plot(x, lpnorm_scaled(x, 1.0, mu), color=colors[1], label=labels[1], linewidth=3)


for i in xrange(muvalues.__len__()):
    mu = muvalues[i]
    plt.plot(x, lpnorm_scaled(x, 0.1, mu), color=colors[i+2], label=labels[i+2], linewidth=3)

plt.legend()
axes = plt.gca()
axes.set_ylim([0,1])
axes.set_xlim([-1,1])
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.tight_layout()

plt.legend(loc="lower left")
plt.savefig('lpnorm_fixedp.pdf', dpi=200)




