import numpy as np
from grslra.problems import RobustSubspaceTracking
from scipy.optimize import check_grad
from grslra.testdata import testdata_rst_static


def cost_y_wrapper(y):
    cost = rst.get_cost(y, "y")
    return cost


def grad_y_wrapper(y):
    grad = rst.get_gradient(y, "y")
    return grad


def cost_U_wrapper(U_in):
    U = np.reshape(U_in, (m, k))
    cost = rst.get_cost(U, "U")
    return cost


def grad_U_wrapper(U_in):
    U = np.reshape(U_in, (m, k))
    tmp = rst.get_gradient(U, "U")
    grad = np.outer(tmp[0], tmp[1])
    return grad.flatten()



m = 100
n = 150
k = 10
rho = 0.1
omegasquared=1e-5

p = 0.1
mu = 0.01

X_0, L_0, S_0, U_0, Y_0 = testdata_rst_static(m, n, k, rho, omegasquared)


# rate_Omega = 0.1
# card_Omega = np.int(np.round(rate_Omega * m * n))
# Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))
#
# X_Omega = X_0[Omega]

rst = RobustSubspaceTracking(m, k, p, mu)
rst.load_sample(X_0[:,0])

rst.y = np.random.rand(k,)

y = np.random.rand(k,)
U, _ = np.linalg.qr(np.random.rand(m, k))

print check_grad(cost_y_wrapper, grad_y_wrapper, y)
print check_grad(cost_U_wrapper, grad_U_wrapper, U.flatten())

#print check_grad(cost_Y, grad_Y, Y.flatten())