from problems import RobustPCA, RobustSLRA
import lpnorm
import smmprod
import numpy as np
from grslra.tools import innerprod
import timeit
import time
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye


class RobustSLRAold(RobustPCA):
    def __init__(self, X, structure, k, p, mu, rho, Omega=None, dimensions=None, samplesize=None, kappa=None, PCA_INIT=False):
        super(RobustSLRAold, self).__init__(X, k, p, mu, Omega=Omega, dimensions=dimensions, SMMPROD=False, PCA_INIT=PCA_INIT, CALC_L=True, samplesize=samplesize, kappa=kappa)

        self.rho = rho
        self.structure = structure
        self.Lambda = np.zeros((self.m, self.n))
        self.grad_Lambda = np.zeros((self.m, self.n))
        self.N = None

    def update_L(self):
        self.L = np.dot(self.U, self.Y)

    def get_data_vector(self):
        return self.structure.vec_via_orth_proj(self.L)

    def update_Lambda(self):
        self.Lambda = self.Lambda + self.rho * (self.L - self.structure.orth_proj(self.L))
        self.grad_Lambda = (self.Lambda - self.structure.orth_proj(self.Lambda)) / self.mn

    def get_structure_penalty(self):
        return np.linalg.norm(self.L - self.structure.orth_proj(self.L), 'fro') / np.sqrt(self.mn)

    def loss_structure(self, L):
        E = (L - self.structure.orth_proj(L))
        loss_lambda = innerprod(self.Lambda, E) / self.mn
        loss_structure = self.rho / 2.0 * np.linalg.norm(E, 'fro') ** 2 / self.mn
        return loss_lambda + loss_structure

    def grad_structure(self, L):
        grad_structure = self.rho * (L - self.structure.orth_proj(L)) / self.mn
        return self.grad_Lambda + grad_structure

    def get_cost(self, X, varname, VERBOSE=None):
        if varname == "U":
            U = X
            Y = self.Y
        elif varname == "Y":
            Y = X
            U = self.U
        else:
            U = self.U
            Y = self.Y
        # compute full L
        L = np.dot(U, Y)
        loss_data = self.loss_data_full(L)
        loss_structure = self.loss_structure(L)
        if VERBOSE is not None:
            print VERBOSE * "\t" + "data loss: ", loss_data
            print VERBOSE * "\t" + "structure loss: ", loss_structure

        loss = loss_data + loss_structure

        if self.kappa is not None:
            loss_energy = self.loss_energy_full(Y, L)
            if VERBOSE is not None:
                print VERBOSE * "\t" + "energy loss: ", loss_energy
            loss += loss_energy

        return loss

    def get_gradient(self, X, varname):
        if varname == "U":
            U = X
            Y = self.Y
        elif varname == "Y":
            Y = X
            U = self.U
        else:
            return False
        # compute full L
        L = np.dot(U, Y)
        grad = self.grad_data_full(L) + self.grad_structure(L)
        if self.kappa is None:
            if varname == "U":
                return np.dot(grad, self.Y.T)
            elif varname == "Y":
                return np.dot(self.U.T, grad)
        else:
            grad += self.grad_energy_term_full(L)
            if varname == "U":
                return np.dot(grad, self.Y.T)
            elif varname == "Y":
                return np.dot(self.U.T, grad) + self.kappa / self.card_Omega_not * Y

class RobustSLRAtest(RobustSLRAold):
    def __init__(self, X, structure, k, p, mu, rho, Omega=None, dimensions=None, samplesize=None, kappa=None, PCA_INIT=False):
        super(RobustSLRAtest, self).__init__(X, structure, k, p, mu, rho, Omega=Omega, dimensions=dimensions, PCA_INIT=PCA_INIT, samplesize=samplesize, kappa=kappa)
        self.x = structure.vec_via_orth_proj(X)

    def loss_data_full(self, L):
        return self.lpnorm(self.x - self.structure.S_pinv.dot(L.flatten(order='F')))

    def grad_data_full(self, L):
        grad = - self.lpnormgrad(self.x - self.structure.S_pinv.dot(L.flatten(order='F')))
        return grad

    def get_gradient(self, X, varname):
        if varname == "U":
            U = X
            Y = self.Y
        elif varname == "Y":
            Y = X
            U = self.U
        else:
            return False
        L = np.dot(U, Y)
        grad_data = self.grad_data_full(L)
        grad_structure = self.grad_structure(L)
        if varname == "U":
            grad_data_tmp = np.reshape(self.structure.S_pinv.dot(spkron(Y.T, speye(self.m))).T.dot(grad_data), (self.m, self.k), order='F')
            grad = grad_data_tmp + np.dot(grad_structure, Y.T)
        elif varname == "Y":
            grad_data_tmp = np.reshape(self.structure.S_pinv.dot(spkron(speye(self.n), U)).T.dot(grad_data), (self.k, self.n), order='F')
            grad = grad_data_tmp + np.dot(U.T, grad_structure)
        return grad

    def loss_structure(self, L):
        l = L.flatten('F')
        e = l - self.structure.S.dot(self.structure.S_pinv.dot(l))
        loss_lambda_vec = np.dot(self.Lambda.flatten('F'), e) / self.mn
        loss_structure_vec = self.rho / 2.0 * np.linalg.norm(e) ** 2 / self.mn
        return loss_lambda_vec + loss_structure_vec

class RobustSLRAcollaborative(RobustSLRAold):

    def __init__(self, X, structure, k, p, mu, rho, Omega=None, dimensions=None, BIGDATA=None, kappa=None):
        self.BIGDATA = BIGDATA
        if dimensions is None:
            dimensions_init = None
        else:
             dimensions_init = dimensions[1:3]

        if Omega is None:
            Omega_init = None
        else:
            Omega_init = Omega[0]

        super(GRSLRACollaborative, self).__init__(X[0], structure, k, p, mu, rho, Omega=Omega_init, dimensions=dimensions_init, kappa=kappa)

        if Omega is None:
            self.X = X
            self.X_Omega = None
            self.Xmax = self.X.max()
            self.Xmin = self.X.min()
            self.N = X.shape[0]
            self.card_Omega = self.N * self.mn
        else:
            self.X = None
            self.X_Omega = np.hstack(X)
            self.Xmax = self.X_Omega.max()
            self.Xmin = self.X_Omega.min()
            self.N = len(X)
            Omega_full = np.vstack([np.hstack([i*np.ones((x[0].size, 1), dtype=np.int), np.atleast_2d(x[0]).T, np.atleast_2d(x[1]).T]) for i,x in enumerate(Omega)])
            self.Omega = (Omega_full[:,0],Omega_full[:,1],Omega_full[:,2])
            self.card_Omega = self.X_Omega.size
            self.Psi = self.Omega
            self.X_Psi = self.X_Omega
            if BIGDATA is not None:
                self.Xlist = X
                self.Omegalist = Omega

    def loss_data_full(self, L):
        if self.Psi is None:
            # compute full residual
            return self.lpnorm(self.X - L)
        else:
            # compute selective entries of the residual
            return self.lpnorm(self.X_Psi - L[self.Psi[1:3]])

    def grad_data_full(self, L):
        if self.Omega is None:
            return -np.sum(self.lpnormgrad(self.X - L), axis=0)
        elif self.BIGDATA is not None:
            grad = np.zeros((self.m, self.n))
            for i in xrange(self.N):
                values = -self.lpnormgrad(self.Xlist[i] - L[self.Omegalist[i]])
                grad[self.Omegalist[i]] += values
                return grad
        else:
            grad = np.zeros((self.N, self.m, self.n))
            values = -self.lpnormgrad(self.X_Omega - L[self.Psi[1:3]])
            grad[self.Psi] = values
        return np.sum(grad, axis=0)

class GRSLRAblock(RobustSLRA):
    def __init__(self, x, structure, k, p, mu, rho, Omega=None, PCA_INIT=False, U_init=None):
        super(RobustSLRA, self).__init__(k, p, mu)
        self.xmin = x.min()
        self.xmax = x.max()

        self.x = None
        self.x_Omega = None
        self.structure = structure
        self.m = structure.a * structure.m
        self.n = structure.b * structure.n
        self.mn = self.m * self.n
        self.N = structure.M * structure.N
        self.rho = rho
        self.Omega = Omega
        self.vec_Lambda = np.zeros((self.mn,))
        self.vec_grad_Lambda = np.zeros((self.mn,))

        if Omega is None:
            # assume x is the full data vector
            self.x = x
            self.x_Omega = None
            if x.size != self.N:
                print "Error: Input length does not match the structure"
                exit()
            self.card_Omega = self.N
        else:
            # assume only the known entries are given
            self.x = None
            self.x_Omega = x
            self.card_Omega = Omega.size

        if PCA_INIT:
            if self.Omega:
                x_full = np.zeros((self.N,))
                x_full[Omega] = x
                X_full = structure.struct_from_vec(x_full)
            else:
                X_full = structure.struct_from_vec(x)

            U, _, _ = np.linalg.svd(X_full)
            self.U = U[:, : k]
            self.Y = np.dot(self.U.T, X_full)
        else:
            if U_init is not None:
                self.U = U_init
            else:
                self.U, _ = np.linalg.qr(np.random.randn(self.m, self.k))
            self.Y = np.zeros((self.k,self.n))

        self.update_L()