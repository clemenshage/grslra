import numpy as np
from numpy import random as npr
from scipy.linalg import expm
from scipy.linalg import orth
from grslra.structures import Hankel
from grslra.tools import incoherence
import smmprod


def testdata_rpca_lmafit(m, n, k, rho, sigma=None, delta=None):
    # This function generates test data according to the data model of LMaFit, see Shen et al. 2014
    V = npr.normal(0, 1, (m, k))
    W = npr.normal(0, 1, (k, n))
    if sigma is None:
        sigma = 0.01
    S = sprandn(m, n, rho, sigma * m)
    L = np.dot(V, W)
    U, R = np.linalg.qr(V)
    Y = np.dot(R, W)
    X = L + S
    if delta is not None:
        N = npr.randn(m, n)
        N *= delta * np.linalg.norm(L, 'fro') / np.linalg.norm(N, 'fro')
        X += N
    return X, L, S, U, Y


def testdata_rpca_lmafit_bigdata(m, n, k, Omega, rho, sigma=None, delta=None):
    # This function generates large-scale test data according to the data model of LMaFit, see Shen et al. 2014
    # Notice that the actual outlier rate may be lower rho due to duplicate index pairs
    card_Omega = Omega[0].size
    V = npr.normal(0, 1, (m, k))
    W = npr.normal(0, 1, (k, n))
    L = smmprod.smmprod_c(V, W, Omega)
    S = np.zeros(card_Omega)
    nnz_S = np.int(np.round(rho * card_Omega))
    ix_S = npr.choice(card_Omega, nnz_S, replace=False)
    if sigma is None:
        sigma = 0.01
    values_S = sigma * m * 2 * npr.standard_normal((nnz_S, ))
    S[ix_S] = values_S
    X = L + S
    if delta is not None:
        N = npr.standard_normal((card_Omega,))
        N *= delta * np.linalg.norm(L) / np.linalg.norm(N)
        X += N
    U, R = np.linalg.qr(V)
    Y = np.dot(R, W)
    return X, L, S, U, Y


def testdata_rpca_ialm(m, n, k, rho, amplitude=None, sigma=None, CHECK_COHERENCE=False):
    # This function generates test data according to the data model of Candes et al. 2011
    U = npr.normal(0, 1.0 / m, (m, k))
    V = npr.normal(0, 1.0 / n, (k, n))
    L = np.dot(U, V)
    U, R = np.linalg.qr(U)
    Y = np.dot(R, V)
    if amplitude is None:
        amplitude = 1
    S = spbernoulli(m, n, rho, (-amplitude, amplitude))
    X = L + S
    if sigma is not None:
        N = sigma * npr.standard_normal((m, n))
        X += N
    if CHECK_COHERENCE:
        mu_U, mu_V = incoherence(L, k)
        print "Left and right bases of L are incoherent with mu(U) = ", mu_U, " and mu(V) = ", mu_V
    return X, L, S, U, Y


def testdata_rst_static(m, n, k, rho, omegasquared):
    U, _ = np.linalg.qr(npr.randn(m, k))
    W = npr.randn(k, n)
    L = np.dot(U, W)
    S = sprandn(m, n, rho, L.max())
    Eta = omegasquared * npr.randn(m, n)

    X = L + S + Eta

    return X, L, S, U, W


def sprandn(m, n, rho, amplitude):
    # This function creates a matrix of density rho, whose randomly placed elements are Gaussian distributed
    S = np.zeros((m, n))
    nzentries = np.int(np.round(rho * m * n))
    values = amplitude * npr.standard_normal((nzentries,))
    Omega = np.unravel_index(npr.choice(m * n, nzentries, replace=False), (m, n))
    S[Omega] = values
    return S


def spbernoulli(m, n, rho, values):
    # This function creates a matrix of density rho, whose randomly placed elements are Bernoulli distributed
    S = np.zeros((m, n))
    nzentries = np.int(np.round(rho * m * n))
    entries = npr.choice(values, nzentries, replace=True)
    Omega = np.unravel_index(npr.choice(m * n, nzentries, replace=False), (m, n))
    S[Omega] = entries
    return S


def testdata_lti(N, m, k):
    n = N - m + 1

    A = np.dot(orth(npr.randn(k, k)), orth(npr.randn(k, k)).T)
    U = np.zeros((m, k))
    Y = np.zeros((k, n))

    tmp = npr.standard_normal(k, )
    U[0, :] = tmp / np.linalg.norm(tmp)
    tmp = npr.standard_normal((k, ))
    Y[:, 0] = tmp / np.linalg.norm(tmp)

    for mx in xrange(1, m):
        U[mx, :] = np.dot(U[mx - 1, :], A)

    for nx in xrange(1, n):
        Y[:, nx] = np.dot(A, Y[:, nx - 1])

    Q, R = np.linalg.qr(U)
    U = Q
    Y = np.dot(R, Y)
    X_H = np.dot(U, Y)

    hankel = Hankel(m, n)
    x = hankel.vec_from_struct(X_H)

    return x, U, Y


def testdata_lti_outliers(N, m, k, rho, amplitude=None, sigma=None):
    x_0, U, Y = testdata_lti(N, m, k)

    N = x_0.shape[0]

    if sigma is not None:
        x = x_0 + sigma * npr.standard_normal((N,))
    else:
        x = x_0

    if amplitude is None:
        amplitude = 1
    s = spbernoulli(1, N, rho, (-amplitude, amplitude))
    s = np.reshape(s, (N,))

    x += s

    return x, x_0, U, Y


def testdata_ltv(N, k, t=0.002):
    x = np.zeros(N, )

    tmp = npr.standard_normal((k, ))
    b = tmp / np.linalg.norm(tmp)
    tmp = npr.standard_normal((k, ))
    c = tmp / np.linalg.norm(tmp)

    Z = npr.randn(k, k)
    Z = Z - Z.T

    z = b
    x[0] = np.dot(c, z)

    for i in range(N - 1):
        A = expm(t * (i + 200) * Z)
        z = np.dot(A, z)
        x[i + 1] = np.dot(c, z)

    return x


def testdata_ltv_outliers(N, k, rho, amplitude=None, sigma=None, t=0.001):
    x_0 = testdata_ltv(N, k, t=t)

    N = x_0.shape[0]

    if sigma is not None:
        x = x_0 + sigma * npr.standard_normal((N,))

    else:
        x = x_0

    if amplitude is None:
        amplitude = 1
    s = spbernoulli(1, N, rho, (-amplitude, amplitude))
    s = np.reshape(s, (N,))

    x += s

    return x, x_0
