import numpy as np
from grslra.problems import RobustSLRA
from scipy.optimize import check_grad
from grslra.testdata import testdata_lti
from grslra.structures import Hankel
from grslra.tools import innerprod


def f_data_Y(Y_in):
    Y = np.reshape(Y_in, (k, n))
    L = np.dot(U, Y)
    loss_data = grslra.lpnorm(X - L)
    return loss_data


def grad_data_Y(Y_in):
    Y = np.reshape(Y_in, (k, n))
    L = np.dot(U, Y)
    grad_data = -np.dot(U.T, grslra.lpnormgrad(X - L))
    return grad_data.flatten()


def f_Lambda_Y(Y_in):
    Y = np.reshape(Y_in, (k, n))
    L = np.dot(U, Y)
    E = (L - hankel.orth_proj(L))
    loss_lambda = -innerprod(Lambda, E) / (m * n)
    return loss_lambda


def grad_Lambda_Y(dummy):
    grad = -np.dot(U.T, (Lambda - hankel.orth_proj(Lambda))/ (m * n))
    return grad.flatten()


def f_structure_Y(Y_in):
    Y = np.reshape(Y_in, (k, n))
    L = np.dot(U, Y)
    E = (L - hankel.orth_proj(L))
    loss_structure = rho / 2.0 * np.linalg.norm(E, 'fro') ** 2 / (m * n)
    return loss_structure


def grad_structure_Y(Y_in):
    Y = np.reshape(Y_in, (k, n))
    L = np.dot(U, Y)
    E = (L - hankel.orth_proj(L))
    grad = rho * np.dot(U.T, E) / (m * n)
    return grad.flatten()


def f_Y(Y_in):
    Y = np.reshape(Y_in, (k, n))
    L = np.dot(U, Y)
    loss_data = grslra.lpnorm(X - L)
    E = (L - hankel.orth_proj(L))
    loss_lambda = -innerprod(Lambda, E) / (m*n)
    loss_structure = rho / 2.0 * np.linalg.norm(E, 'fro') ** 2 / (m*n)
    return loss_data + loss_lambda + loss_structure


def grad_Y(Y_in):
    Y = np.reshape(Y_in, (k, n))
    L = np.dot(U, Y)
    grad_data = -grslra.lpnormgrad(X - L)
    grad_lambda = -(Lambda - hankel.orth_proj(Lambda)) / (m * n)
    grad_structure = rho * (L - hankel.orth_proj(L)) / (m*n)
    grad = np.dot(U.T, (grad_data + grad_lambda + grad_structure))
    return grad.flatten()


def f_data_U(U_in):
    U = np.reshape(U_in, (m, k))
    L = np.dot(U, Y)
    loss_data = grslra.lpnorm(X - L)
    return loss_data


def grad_data_U(U_in):
    U = np.reshape(U_in, (m, k))
    L = np.dot(U, Y)
    grad_data = -np.dot(grslra.lpnormgrad(X - L), Y.T)
    return grad_data.flatten()


def f_Lambda_U(U_in):
    U = np.reshape(U_in, (m, k))
    L = np.dot(U, Y)
    E = (L - hankel.orth_proj(L))
    loss_lambda = -innerprod(Lambda, E) / (m*n)
    return loss_lambda


def grad_Lambda_U(dummy):
    grad = -np.dot((Lambda - hankel.orth_proj(Lambda)) / (m * n), Y.T)
    return grad.flatten()


def f_structure_U(U_in):
    U = np.reshape(U_in, (m, k))
    L = np.dot(U, Y)
    E = (L - hankel.orth_proj(L))
    loss_structure = rho / 2.0 * np.linalg.norm(E, 'fro') ** 2 / (m * n)
    return loss_structure


def grad_structure_U(U_in):
    U = np.reshape(U_in, (m, k))
    L = np.dot(U, Y)
    E = (L - hankel.orth_proj(L))
    grad = rho * np.dot(E, Y.T) / (m * n)
    return grad.flatten()


def f_U(U_in):
    U = np.reshape(U_in, (m, k))
    L = np.dot(U, Y)
    loss_data = grslra.lpnorm(X - L)
    E = (L - hankel.orth_proj(L))
    loss_lambda = -innerprod(Lambda, E) / (m*n)
    loss_structure = rho / 2.0 * np.linalg.norm(E, 'fro') ** 2 / (m*n)
    return loss_data + loss_lambda + loss_structure


def grad_U(U_in):
    U = np.reshape(U_in, (m, k))
    L = np.dot(U, Y)
    grad_data = -grslra.lpnormgrad(X - L)
    grad_lambda = -(Lambda - hankel.orth_proj(Lambda)) / (m * n)
    grad_structure = rho * (L - hankel.orth_proj(L)) / (m*n)
    grad = np.dot((grad_data + grad_lambda + grad_structure), Y.T)
    return grad.flatten()


def cost_Y_wrapper(Y_in):
    Y = np.reshape(Y_in, (k, n))
    cost = grslra.get_cost(Y, "Y")
    return cost


def grad_Y_wrapper(Y_in):
    Y = np.reshape(Y_in, (k, n))
    grad = grslra.get_gradient(Y, "Y")
    return grad.flatten()


def cost_U_wrapper(U_in):
    U = np.reshape(U_in, (m, k))
    cost = grslra.get_cost(U, "U")
    return cost


def grad_U_wrapper(U_in):
    U = np.reshape(U_in, (m, k))
    grad = grslra.get_gradient(U, "U")
    return grad.flatten()


m = 100
N = 200
n = N - m + 1
k = 10

p = 0.5
mu = 0.1
rho = 1.0

hankel = Hankel(m, n)

x, U, Y = testdata_lti(N, m, k)
rate_Omega = 0.5
card_Omega = np.int(np.round(rate_Omega * m * n))
Omega = np.unravel_index(np.random.choice(m * n, card_Omega, replace=False), (m, n))

grslra = RobustSLRA(x, hankel, k, p, mu, rho)

grslra.Y = np.random.rand(k, n)
grslra.update_L()
grslra.update_Lambda()

Lambda=np.random.rand(m, n)

Y = np.random.rand(k,n)
U, _ = np.linalg.qr(np.random.rand(m, k))

# print check_grad(f_data_Y, grad_data_Y, Y.flatten())
# print check_grad(f_Lambda_Y, grad_Lambda_Y, Y.flatten())
# print check_grad(f_structure_Y, grad_structure_Y, Y.flatten())
# print check_grad(f_Y, grad_Y, Y.flatten())
# print check_grad(f_data_U, grad_data_U, U.flatten())
# print check_grad(f_Lambda_U, grad_Lambda_U, U.flatten())
# print check_grad(f_structure_U, grad_structure_U, U.flatten())
# print check_grad(f_U, grad_U, U.flatten())



print check_grad(cost_Y_wrapper, grad_Y_wrapper, Y.flatten())
print check_grad(cost_U_wrapper, grad_U_wrapper, U.flatten())