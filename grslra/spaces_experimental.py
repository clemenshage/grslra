from spaces import SpaceMeta
import numpy as np
from tools import qr_positive


class GrassmannianQR(SpaceMeta):
    # see Hage and Kleinsteuber 2014 for details on the used retraction
    def __init__(self, m, k):
        self.V = np.zeros((m, m))
        self.Theta = np.zeros((m, 2 * k))
        self.R = np.zeros((k, k))
        self.Theta_H = np.zeros((m - k, m - k))

    def get_G(self, grad, X):
        grad = 0.5 * (grad + grad.T)
        m, k = X.shape
        self.V, _ = qr_positive(X, mode='complete')
        X_orth = self.V[:, k:m]
        G = np.dot(X_orth.T, np.dot(grad, X))
        return G

    def get_H(self, G, gamma, tauH):
        m_k, k = G.shape
        H = -G + gamma * tauH
        self.Theta_H, R = qr_positive(H, mode='complete')
        self.R = R[:np.minimum(m_k, k), :]
        self.Theta = np.dot(self.V, np.vstack((np.hstack((np.identity(k), np.zeros((k, k)))),
                                               np.hstack((np.zeros((m_k, k)), self.Theta_H[:, :k])))))
        return H

    def update_variable(self, X, H, t):
        m, k = X.shape
        M = np.vstack((np.hstack((np.identity(k), -t * self.R.T)), np.hstack((t * self.R, np.identity(k)))))
        Theta_M, _ = qr_positive(M, mode='complete')
        return np.dot(self.Theta, Theta_M[:, :k])

    def transport(self, Eta, X, H, t):
        tauX = np.dot(self.Theta_H, Eta)
        return tauX