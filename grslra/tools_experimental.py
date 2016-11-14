import numpy as np
from matplotlib import pyplot as plt
import json


def inpainting_preprocess(x, firstmissing, lastmissing):
    omega = np.ones_like(x)
    omega[firstmissing: lastmissing + 1] = 0
    Omega = np.where(omega)[0]
    x_Omega = x[Omega]
    return x_Omega, Omega


def imagesc(X, number=None, Omega=None, dimensions=None):
    if Omega is not None:
        if dimensions is None:
            print "dimensions missing"
            return 0
        X_Omega = X
        X = np.zeros(dimensions)
        X[Omega] = X_Omega

    plt.ion()
    if number is not None:
        plt.figure(number, figsize=(20, 30))
    else:
        plt.figure(figsize=(20, 30))
    plt.imshow(X, interpolation='nearest')
    plt.draw()
    plt.show()


def randomsampling((m, n), rho):
    A = np.zeros((m * n,))
    A_not = np.zeros((m * n,))
    nnz = int(round(rho * m * n))
    indices = np.random.permutation(np.arange(m * n))
    indices_not = indices[nnz:]
    indices = indices[:nnz]
    A[indices] = 1
    A_not[indices_not] = 1
    A = A.reshape((m, n), order='F')
    A_not = A_not.reshape((m, n), order='F')
    return A, A_not


def pca(X, k):
    U, _, _ = np.linalg.svd(X)
    U = U[:, : k]

    L = np.dot(U, np.dot(U.T, X))

    return L


def subspace_angle_via_qr(A, B):
    A, _ = np.linalg.qr(A)
    B, _ = np.linalg.qr(B)

    A_B_orth = B - np.dot(A, np.dot(A.T, B))

    theta = np.rad2deg(np.arcsin(np.linalg.norm(A_B_orth, 2)))
    return theta


def smmprod(A, B, Omega):
    A_srows = A[Omega[0]]
    B_scols = B.T[Omega[1]]
    return np.sum(A_srows * B_scols, axis=1)


def smmprod_loop(A, B, Omega):
    card_Omega = np.size(Omega[0])
    result = np.zeros(card_Omega)
    for k in xrange(card_Omega):
        result[k] = np.dot(A[Omega[0][k]], B.T[Omega[1][k]])
    return result


def incoherence_U(U, mode="basic"):
    m = U.shape[0]
    k = U.shape[1]
    rownorms_U = np.sqrt(np.sum(U.T * U.T, axis=0))
    if mode == "basic":
        mu_U = np.sqrt(m) * np.amax(rownorms_U) / np.sqrt(k)
    elif mode == "all":
        mu_U = np.sqrt(m) * rownorms_U / np.sqrt(k)
    else:
        mu_U = None
    return mu_U


def card_Psi(confidence, tau, sparsity):
    epsilon = tau / 2
    confidence_single = np.sqrt(confidence)
    alpha = 1 - confidence_single
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return (z / epsilon) ** 2 * sparsity * (1 - sparsity)


def save_params(params, name):
    filename = name + '.json'
    with open(filename, 'w') as f:
        json.dump(params, f)
    return 0


def convert_shrinkage(rho_start, rho_end, mu_start, mu_end, c_rho_slow, c_rho_fast):
    I_slow = np.log(rho_end / rho_start) / np.log(c_rho_slow)
    c_mu_slow = np.exp(np.log(mu_end / mu_start) / I_slow)
    I_fast = np.log(rho_end / rho_start) / np.log(c_rho_fast)
    c_mu_fast = np.exp(np.log(mu_end / mu_start) / I_fast)
    return c_mu_slow, c_mu_fast
