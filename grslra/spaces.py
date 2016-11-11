from abc import ABCMeta, abstractmethod
import numpy as np
from tools import msin, mcos


class SpaceMeta:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_G(self, grad, X):
        # this function returns the projected gradient
        pass

    @abstractmethod
    def get_H(self, G, gamma, tauH):
        # this function computes the search direction from the current projected gradient and the past search direction
        pass

    @abstractmethod
    def update_variable(self, X, H, t):
        # this function updates the variable using the search direction and the step size
        pass

    @abstractmethod
    def transport(self, Eta, X, H, t):
        # this function transports an element along H into the tangent space at distance t
        pass


class Euclidean(SpaceMeta):
    def __init__(self):
        pass

    def get_G(self, grad, X):
        return grad

    def get_H(self, G, gamma, tauH):
        H = -G + gamma * tauH
        return H

    def update_variable(self, X, H, t):
        return X + t * H

    def transport(self, Eta, X, H, t):
        return Eta


class GrassmannianSVD(SpaceMeta):
    # see Edelman et al.
    def __init__(self, m, k):
        self.U_H = np.zeros((m, k))
        self.Sigma_H = np.zeros((k, k))
        self.V_H = np.zeros((k, k))

    def get_G(self, grad, X):
        G = grad - np.dot(X, np.dot(X.T, grad))
        return G

    def get_H(self, G, gamma, tauH):
        H = -G + gamma * tauH
        try:
            self.U_H, sigmas_H, V_H_T = np.linalg.svd(H, full_matrices=False)
            self.Sigma_H[:, :] = np.diag(sigmas_H)
            self.V_H = V_H_T.T
        except:
            print "Problem is badly conditioned. Aborting..."
            exit()
        return H

    def update_variable(self, X, H, t):
        X_t = np.dot(np.dot(X, np.dot(self.V_H, mcos(self.Sigma_H * t))) + np.dot(self.U_H, msin(self.Sigma_H * t)), self.V_H.T)
        return X_t

    def transport(self, Eta, X, H, t):
        tauEta = Eta - np.dot(np.dot(np.dot(X, self.V_H), msin(self.Sigma_H * t)) + np.dot(self.U_H, (np.eye(X.shape[1]) - mcos(self.Sigma_H * t))), np.dot(self.U_H.T, Eta))
        return tauEta


class GrassmannianSVDrank1(SpaceMeta):
    # see GROUSE method by Balzano et al.
    def __init__(self, m, k):
        self.u_H = np.zeros((m,))
        self.alpha = 0
        self.sigma_H = 0
        self.v_H = np.zeros((k,))
        self.beta = 0

    def get_G(self, grad, X):
        G = (grad[0] - np.dot(X, np.dot(X.T, grad[0])), grad[1])
        return G

    def get_H(self, G, gamma, tauH):
        H = (-G[0], G[1])
        self.u_H = H[0] / np.linalg.norm(H[0])
        self.v_H = H[1] / np.linalg.norm(H[1])
        self.sigma_H = np.linalg.norm(H[0]) * np.linalg.norm(H[1])
        return H

    def update_variable(self, X, H, t):
        X_t = X + np.outer((np.cos(self.sigma_H * t) - 1) * np.dot(X, self.v_H) + np.sin(self.sigma_H * t) * self.u_H, self.v_H)
        return X_t

    def transport(self, Eta, X, H, t):
        pass