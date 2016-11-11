from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.linalg import hankel
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix as spmatrix



class StructureMeta:
    __metaclass__ = ABCMeta

    @abstractmethod
    def struct_from_vec(self, x):
        # reads an input vector
        # returns a structured matrix
        pass

    @abstractmethod
    def vec_from_struct(self, X):
        # reads a structured matrix
        # returns the underlying data vector
        pass

    @abstractmethod
    def orth_proj(self, X):
        # reads an unstructured matrix
        # returns its orthogonal projection on the space of structured matrices
        pass

    @abstractmethod
    def vec_via_orth_proj(self, X):
        # reads an unstructured matrix
        # returns the underlying data vector of its orthogonal projection on the space of structured matrices
        pass


class Hankel(StructureMeta):

    def __init__(self, m, n):
        self.m = m
        self.n = n
        N = m + n - 1
        self.N = N
        self.S = np.zeros((m*n, N))

        for i in xrange(m):
            self.S[:,i] = hankel(np.vstack((np.zeros((i, 1)), 1, np.zeros((m - i - 1, 1)))), np.zeros((1, n))).reshape((m*n,), order='F')
        for i in xrange(m, N):
            self.S[:, i] = hankel(np.zeros((m, 1)), np.hstack((np.zeros((1, i+1 - m)), np.ones((1,1)), np.zeros((1, N - i - 1))))).reshape((m*n,), order='F')

        self.S = spmatrix(self.S)
        self.ST = self.S.T
        STSinvdiag = 1.0 / (self.ST.dot(self.S)).diagonal()
        STSinv = lil_matrix((N, N))
        STSinv.setdiag(STSinvdiag)
        self.S_pinv = STSinv.dot(self.ST)
        self.Pi_S = self.S.dot(self.S_pinv)

    def struct_from_vec(self, x):
        X_vec = self.S.dot(x)
        X = np.reshape(X_vec, (self.m, self.n), order='F')
        return X

    def vec_from_struct(self, X):
        return np.concatenate((X[0, :], X[1:, -1]))

    def orth_proj(self, X):
        X_H = np.reshape(self.Pi_S.dot(X.flatten(1)), (self.m, self.n), order='F')
        return X_H

    def vec_via_orth_proj(self, X):
        x_h = self.S_pinv.dot(X.flatten(1))
        return x_h
