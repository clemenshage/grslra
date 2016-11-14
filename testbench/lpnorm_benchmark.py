import numpy as np
import time
from matplotlib import pyplot as plt

import lpnorm

algos = [
    lpnorm.lpnorm_py_simple,
    lpnorm.lpnorm_py_inplace,
    lpnorm.lpnorm_c,
    lpnorm.lpnorm_c_sse,
    lpnorm.lpnorm_c_openmp,
    lpnorm.lpnorm_c_sse_openmp,
]

iterations = 10

msizes = [10, 32, 100, 316, 1000, 3162]
iterations = [10000, 1000, 500, 100, 10, 5]


def test_algorithms(*args):
    ''' Print difference of algorithm results '''

    ref = algos[0](*args)
    for algo in algos[1:]:
        print("  %20s | %f" % (algo.__name__, algo(*args)-ref))


def run_algorithms(*args):
    ''' Measure time of each algorithm '''

    ref = None
    for algo in algos:
        start = time.time()
        for i in range(iterations):
            algo(*args)
        stop = time.time()
        duration = stop-start
        ref = ref or duration
        print("  %20s | %f | %2.2f%%" % (algo.__name__, duration,
                                         100.*duration/ref))


def eval_sizes():
    results = np.zeros((len(algos), len(msizes)))
    i = -1
    for algo in algos:
        print algo.__name__
        i +=1
        for j in range(len(msizes)):
            print msizes[j]
            mu = 0.01
            p = 0.1
            total_time = 0
            for k in range(iterations[j]):
                mat = np.random.rand(msizes[j], msizes[j])
                temp = np.zeros_like(mat)
                start = time.time()
                algo(mat, mu, p, temp)
                stop = time.time()
                total_time += (stop-start)
            print i, j
            results[i, j] = total_time / (k+1)
    return results


np.savez('lpnorm_benchmark.npz', algos=algos, msizes=msizes, res=eval_sizes())
