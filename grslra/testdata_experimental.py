import numpy as np
from numpy import random as npr
from scipy.linalg import expm
from scipy.linalg import orth
from structures import Hankel
from tools import smmprod, incoherence


def sprand(m, n, rho, amplitude):
    S = np.zeros((m,n))
    nzentries = np.int(np.round(rho * m * n))
    values = amplitude * 2 * (npr.rand(nzentries,) - 0.5)
    Omega = np.unravel_index(npr.choice(m * n, nzentries, replace=False),(m, n))
    S[Omega] = values
    return S


def testdata_rpca_cost(m, n, k, rho, amplitude=None, sigma=None, CHECK_COHERENCE=False):
    R = npr.standard_normal((m, n))
    U, sigmas, V = np.linalg.svd(R, full_matrices=False)
    Sigma = np.diag(sigmas[:k])
    U = U[:, :k]
    V = V[:, :k]
    Y = np.dot(Sigma, V)
    L = np.dot(U, Y)
    L = L/(L.flatten(1).std())
    if amplitude is None:
        amplitude = 5
    S = sprand(m, n, rho, amplitude)
    X = L + S
    if sigma is not None:
        N = sigma * npr.standard_normal((m,n))
        X += N
    if CHECK_COHERENCE:
        mu_U, mu_V = incoherence(L, k)
        print "Left and right bases of L are incoherent with mu(U) = ", mu_U, " and mu(V) = ", mu_V
    return X, L, S, U, Y

def testdata_rst_rotating(m, n, k, rho, delta=1e-5, omegasquared=1e-5):
    L = np.zeros((m, n))

    U, _ = np.linalg.qr(npr.randn(m, k))
    B = npr.rand(m, m)
    B -= B.T

    e_tothe_deltaB = expm(delta * B)
    y = npr.rand(k,)
    L[:, 0] = np.dot(U, y)

    for t in xrange(1,n):
        U = np.dot(e_tothe_deltaB, U)
        y = npr.rand(k,)
        L[:, t] = np.dot(U, y)

    S = sprandn(m, n, rho, np.max(L))
    N = omegasquared * npr.randn(m, n)

    X = L + S + N

    return X, L, S