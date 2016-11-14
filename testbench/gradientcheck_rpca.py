import numpy as np
from grslra.problems import RobustPCA
from scipy.optimize import check_grad
from grslra.testdata import testdata_rpca_lmafit


def cost_Y_wrapper(Y_in):
    Y = np.reshape(Y_in, (k, n))
    cost = rpca.get_cost(Y, "Y")
    return cost


def grad_Y_wrapper(Y_in):
    Y = np.reshape(Y_in, (k, n))
    grad = rpca.get_gradient(Y, "Y")
    return grad.flatten()


def cost_U_wrapper(U_in):
    U = np.reshape(U_in, (m, k))
    cost = rpca.get_cost(U, "U")
    return cost


def grad_U_wrapper(U_in):
    U = np.reshape(U_in, (m, k))
    grad = rpca.get_gradient(U, "U")
    return grad.flatten()



m = 100
n = 150
k = 10
rho = 0.1

p = 0.5
mu = 0.1

X_0, L_0, S_0, U_0, Y_0 = testdata_rpca_lmafit(m, n, k, rho)


rate_Omega = 0.1
card_Omega = np.int(np.round(rate_Omega * m * n))
Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))

X_Omega = X_0[Omega]


rpca = RobustPCA(X_Omega, k, p, mu, Omega=Omega, dimensions=(m, n), kappa=1)

rpca.Y = np.random.rand(k, n)
rpca.update()

Y = np.random.rand(k,n)
U, _ = np.linalg.qr(np.random.rand(m, k))

print check_grad(cost_Y_wrapper, grad_Y_wrapper, Y.flatten())
print check_grad(cost_U_wrapper, grad_U_wrapper, U.flatten())

#print check_grad(cost_Y, grad_Y, Y.flatten())