import numpy as np
from matplotlib import pyplot as plt

data = np.load('lpnorm_benchmark.npz')
algos = data["algos"]
msizes = data["msizes"]
res = data["res"]


for i in range(len(algos)):
    plt.loglog(np.square(msizes), res[i, :], linewidth=2.5, linestyle="-", label=algos[i].__name__.replace('_', ' '))

plt.legend(loc="lower right")
plt.show()