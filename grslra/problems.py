import numpy as np
import timeit
import lpnorm
import smmprod
from scipy.sparse import coo_matrix as spmatrix


class RobustAlgo(object):
    # This class implements the smoothed lp-norm loss function and its gradient
    def __init__(self, k, p, mu):
        self.k = k
        self.p = p
        self.mu = mu

    def lpnorm(self, X):
        # Depending on the cardinality of the input, the cost function is either computed in Python or with an external C module
        card = X.size
        if card > 1e4:
            val = lpnorm.lpnorm_c_openmp(X, self.mu, self.p) / card
        else:
            val = lpnorm.lpnorm_py_simple(X, self.mu, self.p) / card
        zero_offset = self.mu ** (self.p / 2.0)
        scaling = (self.mu + 1) ** (self.p / 2.0) - zero_offset
        return (val - zero_offset) / scaling

    def lpnormgrad(self, X):
        grad = lpnorm.lpnormgrad_c_openmp(X, self.mu, self.p) / X.size
        zero_offset = self.mu ** (self.p / 2.0)
        scaling = (self.mu + 1) ** (self.p / 2.0) - zero_offset
        return grad / scaling


class RobustPCA(RobustAlgo):
    def __init__(self, X, k, p, mu, kappa=None, samplesize=None, Omega=None, dimensions=None, PCA_INIT=False, SMMPROD=None, CALC_L=False, U_init=None):
        super(RobustPCA, self).__init__(k, p, mu)

        self.Xmax = np.float(X.max())
        self.Xmin = np.float(X.min())

        self.Omega = Omega
        # X is a matrix, X_Omega is a vector containing the observed entries of X
        self.X = None
        self.X_Omega = None

        if Omega is None:
            # assume X is given as a matrix
            self.X = X
            self.X_Omega = None
            self.m, self.n = X.shape
            self.mn = self.m * self.n
            self.card_Omega = self.mn
        else:
            # assume X is given as a vector
            self.X = None
            self.X_Omega = X
            self.m, self.n = dimensions
            self.mn = self.m * self.n
            self.card_Omega = Omega[0].size

        if samplesize is None:
            self.card_Psi = self.card_Omega
        else:
            # cannot sample more than what is observed
            self.card_Psi = np.int(np.minimum(samplesize, self.card_Omega))

        self.card_Omega_not = self.mn - self.card_Omega
        self.card_Psi_not = self.mn - self.card_Psi

        # kappa is an optional energy factor that weighs an additional cost function term for the energy on the unobserved entries. See RTRMC by Boumal and Absil for the general idea
        if kappa is not None:
            self.kappa = np.float(kappa)
        else:
            self.kappa = None

        # if the external parameter enforces selective matrix-matrix product then use this setting, but only if it is an actual subsampling
        if SMMPROD is not None:
            if self.card_Psi < self.mn:
                self.SMMPROD_PSI = SMMPROD
            else:
                self.SMMPROD_PSI = False
            if self.card_Omega < self.mn:
                self.SMMPROD_OMEGA = SMMPROD
            else:
                self.SMMPROD_OMEGA = False
        else:
            # otherwise run benchmark to measure if SMMPROD is faster than full matrix-matrix product
            if self.card_Psi == self.mn:
                # if all entries are sampled SMMPROD is slow
                self.SMMPROD_PSI = False
            else:
                # run benchmark for Psi
                self.SMMPROD_PSI = self.mmprod_benchmark(self.card_Psi)

            if Omega is None or not self.SMMPROD_PSI:
                # if all entries are sampled SMMPROD is slow
                # since Psi is a subset of Omega, SMMPROD for Omega is slow if SMMPROD for Psi is slow
                self.SMMPROD_OMEGA = False
            else:
                self.SMMPROD_OMEGA = self.mmprod_benchmark(self.card_Omega)

        if self.SMMPROD_PSI:
            print "Using selective matrix multiplication for cost function"
        else:
            print "Using full matrix multiplication for cost function"

        if self.SMMPROD_OMEGA:
            print "Using selective matrix multiplication for gradient"
        else:
            print "Using full matrix multiplication for gradient"

        self.CALC_L = CALC_L

        # initialize U and Y
        if PCA_INIT:
            if self.Omega:
                X_full = np.zeros((self.m, self.n))
                X_full[Omega] = X
                U, _, _ = np.linalg.svd(X_full)
                self.U = U[:, : k]
                self.Y = np.dot(self.U.T, X_full)
            else:
                U, _, _ = np.linalg.svd(self.X)
                self.U = U[:, : k]
                self.Y = np.dot(self.U.T, self.X)
        else:
            if U_init is not None:
                self.U = U_init
            else:
                self.U, _ = np.linalg.qr(np.random.randn(self.m, self.k))
            self.Y = np.zeros((self.k, self.n))

        # Psi is the set of samples for each iteration, it is a subset of Omega
        self.Psi = None
        # ix_Psi denotes the indices of Psi with respect to Omega
        self.ix_Psi = None
        # X_Psi denotes the vector of sampled entries
        self.X_Psi = None
        self.L = None
        self.update()

    def loss_data_selective(self, UY):
        return self.lpnorm(self.X_Psi - UY)

    def loss_data_full(self, L):
        if self.Psi is None:
            # compute full residual
            return self.lpnorm(self.X - L)
        else:
            # compute selective entries of the residual
            return self.lpnorm(self.X_Psi - L[self.Psi])

    def grad_data_selective(self, UY):
        values = -self.lpnormgrad(self.X_Omega - UY)
        return spmatrix((values, self.Omega), shape=(self.m, self.n))

    def grad_data_full(self, L):
        if self.Omega is None:
            grad = -self.lpnormgrad(self.X - L)
        else:
            grad = np.zeros((self.m, self.n))
            grad[self.Omega] = -self.lpnormgrad(self.X_Omega - L[self.Omega])
        return grad

    # loss and gradient functions for additional energy term
    def loss_energy_full(self, Y, L):
        return self.kappa / 2.0 * (np.linalg.norm(Y, 'fro') ** 2 - np.linalg.norm(L[self.Omega]) ** 2) / self.card_Omega_not

    def loss_energy_selective(self, Y, UY):
        return self.kappa / 2.0 * (np.linalg.norm(Y, 'fro') ** 2 - np.linalg.norm(UY) ** 2) / self.card_Psi_not

    def grad_energy_term_selective(self, UY):
        values = self.kappa * UY / self.card_Psi_not
        return spmatrix((values, self.Omega), shape=(self.m, self.n))

    def grad_energy_term_full(self, L):
        grad = np.zeros((self.m, self.n))
        grad[self.Omega] = self.kappa * L[self.Omega] / self.card_Omega_not
        return grad

    def get_cost(self, var, varname, VERBOSE=None):
        if varname == "U":
            U = var
            Y = self.Y
        elif varname == "Y":
            Y = var
            U = self.U
        else:
            U = self.U
            Y = self.Y
        if self.kappa is not None:
            if self.SMMPROD_PSI:
                # compute selective entries of the low-rank approximation
                UY = smmprod.smmprod_c(U, Y, self.Psi)
                loss_data = self.loss_data_selective(UY)
                loss_energy = self.loss_energy_selective(Y, UY)
            else:
                # compute full L
                L = np.dot(U, Y)
                loss_data = self.loss_data_full(L)
                loss_energy = self.loss_energy_full(Y, L)
            if VERBOSE is not None:
                print VERBOSE * "\t" + "loss_data: ", loss_data
                print VERBOSE * "\t" + "loss_energy: ", loss_energy
            loss = loss_data + loss_energy
        else:
            if self.SMMPROD_PSI:
                # compute selective entries of the low-rank approximation
                UY = smmprod.smmprod_c(U, Y, self.Psi)
                loss_data = self.loss_data_selective(UY)
            else:
                # compute full L
                L = np.dot(U, Y)
                loss_data = self.loss_data_full(L)
            if VERBOSE is not None:
                print VERBOSE * "\t" + "loss_data: ", loss_data
            loss = loss_data
        return loss

    def get_full_cost(self):
        # this function computes the full cost on all observable positions
        if self.card_Psi == self.card_Omega:
            cost = self.get_cost(None, "full")
        else:
            Psi_backup = self.Psi
            X_Psi_backup = self.X_Psi
            SMMPROD_PSI_backup = self.SMMPROD_PSI

            self.Psi = self.Omega
            self.X_Psi = self.X_Omega
            self.SMMPROD_PSI = self.SMMPROD_OMEGA
            cost = self.get_cost(None, "full")
            self.Psi = Psi_backup
            self.X_Psi = X_Psi_backup
            self.SMMPROD_PSI = SMMPROD_PSI_backup
        return cost

    def get_gradient(self, var, varname):
        if varname == "U":
            U = var
            Y = self.Y
        elif varname == "Y":
            Y = var
            U = self.U
        else:
            return False
        if self.SMMPROD_OMEGA:
            # compute selective entries of L
            UY = smmprod.smmprod_c(U, Y, self.Omega)
            grad_data = self.grad_data_selective(UY)
            if self.kappa is not None:
                if varname == "U":
                    return (grad_data - self.grad_energy_term_selective(UY)).dot(Y.T)
                elif varname == "Y":
                    return (grad_data - self.grad_energy_term_selective(UY)).T.dot(U).T + self.kappa/self.card_Omega_not * Y
                else:
                    return False
            else:
                if varname == "U":
                    return grad_data.dot(self.Y.T)
                elif varname == "Y":
                    return grad_data.T.dot(self.U).T
                else:
                    return False
        else:
            # compute full L
            L = np.dot(U, Y)
            grad_data = self.grad_data_full(L)
            if self.kappa is not None:
                if varname == "U":
                    return np.dot(grad_data - self.grad_energy_term_full(L), Y.T)
                elif varname == "Y":
                    return np.dot(U.T, grad_data - self.grad_energy_term_full(L)) + self.kappa / self.card_Omega_not * Y
                else:
                    return False
            else:
                if varname == "U":
                    return np.dot(grad_data, self.Y.T)
                elif varname == "Y":
                    return np.dot(self.U.T, grad_data)
                else:
                    return False

    def get_variable(self, varname):
        if varname == "Y":
            return self.Y
        elif varname == "U":
            return self.U
        else:
            return False

    def set_updated(self, var, varname):
        if varname == "U":
            self.U = var
        elif varname == "Y":
            self.Y = var
        else:
            return False

    def update(self):
        self.resample()
        if self.CALC_L:
            self.L = np.dot(self.U, self.Y)

    def resample(self):
        if self.Omega is None:
            # check if all entries are sampled, then no sampling is required and Psi remains None
            if self.card_Psi == self.mn:
                return
            else:
                # randomly pick from all possible positions
                if (self.mn < 1e6) and (self.card_Psi > 0.1 * self.mn):
                    # sample exactly if necessary (large amount of samples, thus high risk of duplicates) and cheap (dimensions small enough)
                    self.Psi = np.unravel_index(np.random.choice(self.mn, self.card_Psi, replace=False), (self.m, self.n))
                else:
                    # sample inexactly, might produce duplicate entries
                    self.Psi = (np.random.choice(self.m, self.card_Psi, replace=True), np.random.choice(self.n, self.card_Psi,  replace=True))
            # X is a matrix since no Omega is defined
            self.X_Psi = self.X[self.Psi]
        else:
            # randomly pick from the entries of Omega
            if self.card_Psi == self.card_Omega:
                self.Psi = self.Omega
                self.X_Psi = self.X_Omega
            else:
                # select random subset
                self.ix_Psi = np.random.choice(self.card_Omega, self.card_Psi, replace=False)
                self.Psi = (self.Omega[0][self.ix_Psi], self.Omega[1][self.ix_Psi])
                self.X_Psi = self.X_Omega[self.ix_Psi]

    def mmprod_benchmark(self, card_selection):
        # This function benchmarks the actual difference in computation time between full and selective matrix-matrix product
        randset = np.unravel_index(np.random.choice(self.mn, card_selection, replace=False), (self.m, self.n))
        A = np.random.rand(self.m, self.k)
        B = np.random.rand(self.k, self.n)

        if self.mn <= 1E5:
            repetitions = 1000
        elif self.mn <= 1E6:
            repetitions = 10
        else:
            repetitions = 1

        time_mmprod = timeit.timeit(lambda: np.dot(A, B), number=repetitions)
        time_smmprod = timeit.timeit(lambda: smmprod.smmprod_c(A, B, randset), number=repetitions)
        # return True if SMMPROD is faster, False if full matrix product is faster
        return time_mmprod > time_smmprod

    def print_cost(self, tablevel):
        self.get_cost(None, "full", VERBOSE=tablevel)


class RobustSubspaceTracking(RobustAlgo):
    def __init__(self, m, k, p, mu, U_init=None):
        super(RobustSubspaceTracking, self).__init__(k, p, mu)
        self.m = m
        self.x = None
        self.x_Omega = None

        if U_init is not None:
            self.U = U_init
        else:
            self.U, _ = np.linalg.qr(np.random.rand(self.m, self.k))
        self.y = np.zeros((k,))
        self.l = np.dot(self.U, self.y)
        self.Omega = None

    def load_sample(self, x, Omega=None):
        # x is always 1D, regardless of Omega, but it is safer to make a distinction between x and x_Omega
        if Omega is not None:
            self.Omega = Omega
            self.x_Omega = x
            self.x = None
        else:
            self.x = x
            self.Omega = None
            self.x_Omega = None

    def loss_data(self, l):
        if self.Omega is None:
            return self.lpnorm(self.x - l)
        else:
            return self.lpnorm(self.x_Omega - l[self.Omega])

    def grad_data(self, l):
        if self.Omega is None:
            grad = -self.lpnormgrad(self.x - l)
        else:
            grad = np.zeros((self.m, ))
            grad[self.Omega] = -self.lpnormgrad(self.x_Omega - l[self.Omega])
        return grad

    def get_variable(self, varname):
        if varname == "y":
            return self.y
        if varname == "U":
            return self.U
        else:
            return False

    def get_gradient(self, var, varname):
        if varname == "y":
            y = var
            l = np.dot(self.U, y)
            return np.dot(self.U.T, self.grad_data(l))
        elif varname == "U":
            U = var
            l = np.dot(U, self.y)
            # Returns a tuple of two vectors instead of actually multiplying out the rank 1 matrix!
            return self.grad_data(l), self.y
        else:
            return False

    def get_cost(self, var, varname, VERBOSE=None):
        if varname == "y":
            y = var
            U = self.U
        elif varname == "U":
            y = self.y
            U = var
        else:
            y = self.y
            U = self.U
        loss_data = self.loss_data(np.dot(U, y))
        if VERBOSE is not None:
            print VERBOSE * "\t" + "loss_data: ", loss_data
        return loss_data

    def set_updated(self, x, varname):
        if varname == "U":
            self.U = x
        elif varname == "y":
            self.y = x
        else:
            return False


class RobustSLRA(RobustAlgo):
    def __init__(self, x, structure, k, p, mu, rho, Omega=None, PCA_INIT=False, U_init=None, Y_init=None):
        super(RobustSLRA, self).__init__(k, p, mu)
        self.xmin = x.min()
        self.xmax = x.max()

        self.x = None
        self.x_Omega = None
        self.structure = structure
        # The structure contains the dimensions of the structured matrix, so no additional dimensions variable is needed
        self.m = structure.m
        self.n = structure.n
        self.mn = self.m * self.n
        # the length of the data vector
        self.N = structure.N
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
            # assume only the known entries of x are given
            self.x = None
            self.x_Omega = x
            self.card_Omega = Omega.size

        if PCA_INIT:
            # if one wishes to initialize with PCA, the full structured matrix needs to be computed first
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
            if Y_init is not None:
                self.Y = Y_init
            else:
                self.Y = np.zeros((self.k, self.n))

        self.L = None
        self.vec_L = None
        self.l = None
        # Since the full L appears in vectorized form in the structural constraint it needs to be stored
        self.update_L()

    def update_L(self):
        self.L = np.dot(self.U, self.Y)
        self.vec_L = self.L.flatten('F')
        self.l = self.structure.S_pinv.dot(self.vec_L)

    def update_Lambda(self):
        self.vec_Lambda += self.rho * (self.vec_L - self.structure.S.dot(self.structure.S_pinv.dot(self.vec_L)))
        self.vec_grad_Lambda = (self.vec_Lambda - self.structure.S.dot(self.structure.S_pinv.dot(self.vec_Lambda))) / self.mn

    def loss_data(self, vec_L):
        # The residual is only computed on the data vector level, not over the full matrix
        if self.Omega is None:
            return self.lpnorm(self.x - self.structure.S_pinv.dot(vec_L))
        else:
            return self.lpnorm(self.x_Omega - self.structure.S_pinv.dot(vec_L)[self.Omega])

    def grad_data_term(self, vec_L):
        if self.Omega is None:
            grad = - self.lpnormgrad(self.x - self.structure.S_pinv.dot(vec_L))
        else:
            grad = np.zeros((self.N,))
            grad[self.Omega] = -self.lpnormgrad(self.x_Omega - self.structure.S_pinv.dot(vec_L)[self.Omega])
        return grad

    def vec_grad_data(self, vec_L):
        if self.Omega is None:
            lpnormgrad = self.lpnormgrad(self.x - self.structure.S_pinv.dot(vec_L))
        else:
            lpnormgrad = np.zeros((self.N,))
            lpnormgrad[self.Omega] = self.lpnormgrad(self.x_Omega - self.structure.S_pinv.dot(vec_L)[self.Omega])

        return -self.structure.S_pinv.T.dot(lpnormgrad)

    def loss_structure(self, vec_L):
        e = vec_L - self.structure.S.dot(self.structure.S_pinv.dot(vec_L))
        loss_lambda_vec = np.dot(self.vec_Lambda, e) / self.mn
        loss_structure_vec = self.rho / 2.0 * np.linalg.norm(e) ** 2 / self.mn
        return loss_lambda_vec + loss_structure_vec

    def vec_grad_structure(self, vec_L):
        vec_grad_structure = self.rho * (vec_L - self.structure.S.dot(self.structure.S_pinv.dot(vec_L))) / self.mn
        return self.vec_grad_Lambda + vec_grad_structure

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
        vec_L = np.dot(U, Y).flatten('F')
        loss_data = self.loss_data(vec_L)
        loss_structure = self.loss_structure(vec_L)
        if VERBOSE is not None:
            print VERBOSE * "\t" + "data loss: ", loss_data
            print VERBOSE * "\t" + "structure loss: ", loss_structure
        loss = loss_data + loss_structure
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
        vec_L = np.dot(U, Y).flatten('F')

        grad = np.reshape(self.vec_grad_data(vec_L) + self.vec_grad_structure(vec_L), (self.m, self.n), order='F')
        if varname == "U":
            return np.dot(grad, self.Y.T)
        elif varname == "Y":
            return np.dot(self.U.T, grad)

    def get_variable(self, varname):
        if varname == "Y":
            return self.Y
        elif varname == "U":
            return self.U
        else:
            return False

    def set_updated(self, var, varname):
        if varname == "U":
            self.U = var
        elif varname == "Y":
            self.Y = var
        else:
            return False

    def print_cost(self, tablevel):
        self.get_cost(None, "full", VERBOSE=tablevel)
