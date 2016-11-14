from structures import StructureMeta
from os import makedirs, path
import numpy as np
from scipy.linalg import hankel
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse import csr_matrix as spmatrix
from structures import Hankel


class BlockHankel(StructureMeta):

    def __init__(self, m, n, a, b, memsave=False):
        # creates an am x bn block Hankel matrix with a x b blocks of m x n Hankel matrices
        # the underlying data vectors form an M x N image patch
        self.m = m
        self.n = n
        M = m + n - 1
        self.a = a
        self.b = b
        N = a + b - 1
        self.M = M
        self.N = N

        nentries = a * b * m * n

        if not path.exists("structfiles"):
            makedirs("structfiles")

        structfilename = path.join("structfiles","blockhankel_" + str(a) + "_" + str(b) + "_" + str(m) + "_" + str(n) + ".npz")

        # check if Structure already exists as a file
        if path.isfile(structfilename):
            print "Loading structure file."
            data = np.load(structfilename)
            self.S = spmatrix((data["data"], (data["row"], data["col"])), shape=data["shape"])
        else:
            print "No structure file found."
            if memsave:
                self.S = lil_matrix((nentries, M * N))
            else:
                self.S = np.zeros((nentries, M * N))

            shaping = self.S[:, 1].shape

            for j in xrange(a):
                A = hankel(np.vstack((np.zeros((j, 1)), 1, np.zeros((a - j - 1, 1)) )), np.zeros((1, b)))
                for i in xrange(m):
                    self.S[:, j * M + i] = np.kron(A, hankel(np.vstack((np.zeros((i, 1)), 1, np.zeros((m - i - 1, 1)))), np.zeros((1, n)))).reshape(shaping, order='F')
                for i in xrange(m, M):
                    self.S[:, j * M + i] = np.kron(A, hankel(np.zeros((m, 1)), np.hstack((np.zeros((1, i+1 - m)), np.ones((1,1)), np.zeros((1, M - i - 1)))))).reshape(shaping, order='F')

                print "Creating Structure...", np.round(np.int(100.0 * np.double(j)/np.double(N))), "% completed"

            for j in xrange(a, N):
                A = hankel(np.zeros((a, 1)), np.hstack((np.zeros((1, j + 1 - a)), np.ones((1,1)), np.zeros((1, N - j - 1)))))

                for i in xrange(m):
                    self.S[:, j * M + i] = np.kron(A, hankel(np.vstack((np.zeros((i, 1)), 1, np.zeros((m - i - 1, 1)))), np.zeros((1, n)))).reshape(shaping, order='F')
                for i in xrange(m, M):
                    self.S[:, j * M + i] = np.kron(A, hankel(np.zeros((m, 1)), np.hstack((np.zeros((1, i+1 - m)), np.ones((1,1)), np.zeros((1, M - i - 1)))))).reshape(shaping, order='F')

                print "Creating Structure...", np.round(np.int(100.0 * np.double(j)/np.double(N))), "% completed"
            self.S = coo_matrix(self.S)
            print "Saving structure"
            np.savez(structfilename, data=self.S.data, col=self.S.col, row=self.S.row, shape=self.S.shape)
            self.S = spmatrix(self.S)

        self.ST = self.S.T
        self.fancy_ix = np.int32(np.reshape(self.S.dot(range(self.S.shape[1])), (self.a * self.m, self.b * self.n), order='F'))

        STSinvdiag = 1.0 / (self.ST.dot(self.S)).diagonal()
        STSinv = lil_matrix((M*N, M*N))
        STSinv.setdiag(STSinvdiag)
        self.S_pinv = STSinv.dot(self.ST)
        self.Pi_S = self.S.dot(self.S_pinv)

        self.STSdiaginv = 1.0 / (self.ST.dot(self.S)).diagonal()

    def struct_from_vec(self, x):
        X_vec = self.S.dot(x)
        X = np.reshape(X_vec, (self.a * self.m, self.b * self.n), order='F')

        #X = x[self.fancy_ix]
        return X

    def vec_from_struct(self, X):
        m = self.m
        n = self.n
        a = self.a
        b = self.b
        N = a + b -1
        M = m + n - 1

        x = np.zeros(M * N,)

        hankel_inner = Hankel(m, n)

        for i in xrange(b):
            H = X[0:m, n * i : n * (i + 1)]
            x[M * i: M * (i+1)] = hankel_inner.vec_from_struct(H)

        for i in xrange(1, a):
            H = X[m * i: m * (i + 1), n * (b-1) : n * b]
            x[M * (i + b - 1): M * (i + b)] = hankel_inner.vec_from_struct(H)

        return x

    def orth_proj(self, X):
        # x_h = np.linalg.lstsq(self.S, X.flatten(1))[0]
        # x_h = lsqr(self.S.T.dot(self.S), self.S.T.dot(X.flatten(1)))[0]
        x_h = self.ST.dot(X.flatten(1)) * self.STSdiaginv

        X_H = self.struct_from_vec(x_h)

        return X_H

    def vec_via_orth_proj(self, X):
        x_h = self.ST.dot(X.flatten(1)) * self.STSdiaginv

        return x_h


class SkewSymmetric(StructureMeta):

    def __init__(self, m):
        self.m = m
        self.N = (m * (m - 1)) / 2
        self.S = lil_matrix((m * m, self.N))

        count = 0
        for i in xrange(m):
            for j in xrange(i + 1, m):
                self.S[i + j * m, count] = 1
                self.S[j + i * m, count] = -1
                count += 1

        self.S = spmatrix(self.S)
        self.ST = self.S.T
        self.S_pinv = 0.5 * self.ST

    def struct_from_vec(self, x):
        X_vec = self.S.dot(x)
        X = np.reshape(X_vec, (self.m, self.m), order='F')
        return X

    def vec_from_struct(self, X):
        x = np.zeros(self.m * (self.m - 1) / 2)
        count = 0
        for i in xrange(self.m):
            for j in xrange(i+1, self.m):
                x[count] = X[i, j]
        return x

    def orth_proj(self, X):
        x = self.S_pinv.dot(X.flatten('F'))
        X_S = self.struct_from_vec(x)
        return X_S

    def vec_via_orth_proj(self, X):
        return self.S_pinv.dot(X.flatten('F'))


class ConcatenatedSkewSymmetric(StructureMeta):

    def __init__(self, m, N):
        self.m = m
        n = N * m
        self.n = n
        self.S = np.zeros((m * n, m * (m - 1) / 2))

        count = -1
        for i in xrange(m):
            for j in xrange(i+1, m):
                tmp = np.zeros((m,m))
                tmp[i , j] = 1
                tmp[j, i] = -1
                count +=1
                self.S[:, count] =  np.tile(tmp,(1,N)).flatten('F')

        self.S_pinv = spmatrix(np.linalg.pinv(self.S))
        self.S = spmatrix(self.S)
        self.ST = self.S.T

    def struct_from_vec(self, x):
        X_vec = self.S.dot(x)
        X = np.reshape(X_vec, (self.m, self.n), order='F')
        return X

    def vec_from_struct(self, X):
        x = np.zeros(self.m * (self.m - 1) / 2)
        count = 0
        for i in xrange(self.m):
            for j in xrange(i+1, self.m):
                x[count] = X[i, j]
        return x

    def orth_proj(self, X):
        x = self.S_pinv.dot(X.flatten('F'))
        X_S = self.struct_from_vec(x)
        return X_S

    def vec_via_orth_proj(self, X):
        return self.S_pinv.dot(X.flatten('F'))