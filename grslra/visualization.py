from time import sleep
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os


def plot_matrices(fignr, layout, *argv, **kwargs):

    if "filename" not in kwargs:
        filename = None
    else:
        filename = kwargs["filename"]

    if "visible" not in kwargs:
        visible = True
    else:
        visible = kwargs["visible"]

    if len(argv) % 3:
        print "Error: False number of arguments given."
        print "Usage: plot_matrices(fignr, layout, filename='auto', Matrix1, 'name1', range1)"
        return -1

    nmatrices = len(argv) / 3

    rows, cols = layout

    plt.figure(fignr, figsize=(cols*3, rows*3))

    if visible:
        plt.ion()
    else:
        plt.ioff()

    for i in xrange(nmatrices):
        data = argv[3 * i]
        title = argv[3 * i + 1]
        range = argv[3 * i + 2]

        plt.subplot(rows, cols, i+1)
        plt.title(title)
        if isinstance(range,tuple):
            plt.imshow(data, interpolation='nearest', vmin=range[0], vmax=range[1])
        else:
            plt.imshow(data, interpolation='nearest')
        plt.gray()
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

    if filename is not None:
        if filename=="auto":
            plt.savefig('/tmp/figure' + str(fignr) + '.png', dpi=100)
        else:
            checkmakedir(filename)
            plt.savefig(filename, dpi=200)

    if not visible:
        plt.close()


def plot_series(fignr, *argv, **kwargs):

    if "filename" not in kwargs:
        filename = None
    else:
        filename = kwargs["filename"]

    if "visible" not in kwargs:
        visible = True
    else:
        visible = kwargs["visible"]

    if len(argv) % 2:
        print "Error: False number of arguments given."
        print "Usage: plot_series(fignr, Series1, 'linestyle')"
        return -1

    nseries = len(argv) / 2

    plt.figure(fignr, figsize=(15,5))
    if visible:
        plt.ion()
    else:
        plt.ioff()

    plt.hold(True)

    nmax=0
    for i in xrange(nseries):
        data = argv[2 * i]
        linestyle = argv[2 * i + 1]
        n = data.shape[0]
        if n > nmax:
            nmax = n
        plt.plot(np.arange(n), data, linestyle)


    if "xlim" in kwargs:
        plt.xlim(kwargs["xlim"])
    else:
        plt.xlim((0, nmax))
    plt.hold(False)

    if filename is not None:
        if filename=="auto":
            plt.savefig('/tmp/figure' + str(fignr) + '.png', dpi=100)
        else:
            checkmakedir(filename)
            plt.savefig(filename, dpi=200)

    if not visible:
        plt.close()


def phaseplot(data, clim, values, maxval, filename):
    matplotlib.rcParams.update({'font.size': 10})
    values = np.double(values)
    fig=plt.figure(1,dpi=300,figsize=(5,5))
    plt.imshow(data, interpolation='nearest', origin='lower')
    plt.set_cmap('gray_r')
    plt.clim(clim)
    plt.xticks(np.arange(values/(10*maxval) - 1, values + 1, values/(10*maxval)),np.arange(0.1,maxval + 0.1,0.1))
    plt.yticks(np.arange(values/(10*maxval) - 1, values + 1, values/(10*maxval)),np.arange(0.1,maxval + 0.1,0.1))
    plt.ylabel(r'$k / m$')
    plt.xlabel(r'$\rho$')
    checkmakedir(filename)
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_lpnorm(p, mu_start, mu_end):
    plt.figure()
    x = np.linspace(-1,1,1001)
    y_0 = lpnorm_scaled(x, p, mu_start)
    y_I = lpnorm_scaled(x, p, mu_end)
    plt.plot(x,y_0, label="first iteration")
    plt.plot(x,y_I, label="last iteration")
    plt.ion()
    plt.legend()
    plt.show()
    sleep(1)
    plt.close()


def lpnorm_scaled(x, p, mu):
    return (lpnorm(x, p, mu) - lpnorm(0, p, mu)) / (lpnorm(1, p, mu) - lpnorm(0, p, mu))


def lpnorm(x, p, mu):
    return (mu + x * x) ** (p / 2.0)


def checkmakedir(filename):
    dir = os.path.dirname(filename)

    if not os.path.exists(dir):
        os.makedirs(dir)