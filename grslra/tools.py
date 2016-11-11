import numpy as np
import json
import os


def innerprod(A, B):
    # This function computes the standard inner product between two matrices via vectorization
    if isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        return np.dot(A.flatten(), B.flatten())
    elif isinstance(A, tuple) and isinstance(B, tuple):
        return np.dot(A[0].flatten(), B[0].flatten()) * np.dot(A[1].flatten(), B[1].flatten())
    else:
        return False


def qr_positive(A, mode='reduced'):
    # This function computes a QR decomposition of a matrix with positive pivot elements in R
    (m, n) = A.shape

    Q, R = np.linalg.qr(A, mode=mode)

    if n > 1:
        dvec = np.sign(np.diag(R))
    else:
        dvec = np.sign(R[0, 0])

    dvec = dvec[:, np.newaxis]

    d = dvec.shape[0]

    R[0: d, :] = R[0: d, :] * dvec
    Q[:, 0: d] = Q[:, 0: d] * dvec.T
    return Q, R


def mcos(X):
    Y = np.zeros(X.shape)
    d = np.diag(X)
    d = np.cos(d)
    Y[:, :] = np.diag(d)
    return Y


def msin(X):
    Y = np.zeros(X.shape)
    d = np.diag(X)
    d = np.sin(d)
    Y[:, :] = np.diag(d)
    return Y


def subspace_angle(U_A, U_B):
    U_A_U_B_orth = U_B - np.dot(U_A, np.dot(U_A.T, U_B))
    theta = np.rad2deg(np.arcsin(np.minimum(np.linalg.norm(U_A_U_B_orth, 2), 1.0)))
    return theta


def rmse(U_0, Y_0, U, Y):
    m = U.shape[0]
    n = Y.shape[1]
    _, R1 = np.linalg.qr(np.hstack((U_0, U)))
    _, R2 = np.linalg.qr(np.hstack((Y_0.T, -Y.T)))
    return np.linalg.norm(np.dot(R1, R2.T), 'fro') / np.sqrt(m * n)


def orthogonality_check(U):
    R = None
    diff1 = np.abs(1.0 - np.dot(U[:, 0], U[:, 0]))
    diff2 = np.abs(np.dot(U[:, 0], U[:, 1]))
    if diff1 > 1e-12 or diff2 > 1e-12:
        # print "Re-orthogonalizing U"
        U, R = qr_positive(U)
    return U, R


def incoherence(L, k):
    m = L.shape[0]
    n = L.shape[1]
    U, _, V_T = np.linalg.svd(L, full_matrices=False)

    U = U[:, :k]
    V = V_T.T[:, :k]

    rownorms_U = np.sqrt(np.sum(U.T * U.T, axis=0))
    mu_U = np.sqrt(m) * np.amax(rownorms_U) / np.sqrt(k)

    rownorms_V = np.sqrt(np.sum(V.T * V.T, axis=0))
    mu_V = np.sqrt(n) * np.amax(rownorms_V) / np.sqrt(k)

    return mu_U, mu_V


def load_params(name):
    basefolder, _ = os.path.split(os.path.abspath(os.path.join(__file__, os.pardir)))
    filename = basefolder + os.sep + 'params' + os.sep + name + '.json'
    with open(filename, 'r') as f:
        params = json.load(f)
    for key, value in params.iteritems():
        if value == "None" or value == "none":
            params[key] = None
        if value == "False" or value == "false":
            params[key] = False
        if value == "True" or value == "true":
            params[key] = True
    return params


def parse_params(params, defaultfile):
    params_default = load_params(defaultfile)
    if params is None:
        return params_default
    else:
        for key in params_default:
            if key not in params:
                params[key] = params_default[key]
    return params


def forecasting_preprocess(x, m, r):
    omega = np.ones_like(x)
    omega = np.hstack((omega, np.zeros(r, )))
    N = omega.size
    Omega = np.where(omega)[0]
    n = N - m + 1
    x_Omega = x[Omega]
    return x_Omega, Omega, n
