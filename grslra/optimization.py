import numpy as np
from grslra.tools import innerprod


class GD(object):
    # This class implements a simple Gradient Descent method
    def __init__(self, problem, varname, space, params):
        self.problem = problem  # the problem we want to solve
        self.varname = varname  # the name of the variable that should be optimized
        self.space = space  # the space in which the variable lives
        self.params = params
        self.G = None # gradient
        self.H = None # search direction
        self.X_old = None
        self.t = None # step size
        self.cost = None
        self.cost_old = None

    def solve(self, initval=None):
        if initval is None:
            self.X = self.problem.get_variable(self.varname)
        else:
            self.X = initval
        self.cost = self.problem.get_cost(self.X, self.varname)
        for i in xrange(self.params["I_max"]):
            self.X_old = self.X
            self.cost_old = self.cost
            self.prepare(i)

            # every 'reset_rate' step initialize with t_init, otherwise initialize close to former step size
            if self.params["step_reset_rate"]==0 or not (i % self.params["step_reset_rate"]):
                self.t = self.params["t_init"]
            else:
                self.t = np.minimum(10 * self.t, self.params["t_init"])
            self.t, X_t, cost_t = self.step()

            if self.t is not None:
                self.X = X_t
                self.cost = cost_t

            if "VERBOSE" in self.params and self.params["VERBOSE"] > 2 and self.t is not None:
                print "\t\t\tOptimizing ", self.varname, " Iteration #", i, ", cost: ", '{:2.6f}'.format(
                    self.cost), ", stepsize: ", '{:.2e}'.format(self.t)

            if self.abort_check():
                break

        self.problem.set_updated(self.X, self.varname)
        return self.cost

    def prepare(self, iteration):
        self.prepare_grad()
        self.H = self.space.get_H(self.G, 0.0, 0.0)

    def prepare_grad(self):
        grad = self.problem.get_gradient(self.X, self.varname)
        self.G = self.space.get_G(grad, self.X)

    def step(self):
        # Simple update with fixed step size
        X_t = self.space.update_variable(self.X, self.H, self.t)
        cost_t = self.problem.get_cost(X_t, self.varname)
        return self.t, X_t, cost_t

    def abort_check(self):
        return False


class GDexact(GD):
    # This class extends the Gradient Descent method above by a Line Search method with Armijo condition
    def __init__(self, problem, varname, space, params):
        super(GDexact, self).__init__(problem, varname, space, params)
        self.GH = None
        self.rho = self.params["rho"]

    def prepare(self, iteration):
        self.prepare_grad()
        self.H = self.space.get_H(self.G, 0.0, 0.0)
        self.GH = innerprod(self.G, self.H)

    def step(self):
        cost = self.cost

        t = self.t / self.rho # needs to be done as step size is multiplied with rho before being used

        t_opt = None  # return None if anything goes wrong
        X_opt = None
        cost_opt = None
        while t > self.params["t_min"]:
            t *= self.rho
            X_t = self.space.update_variable(self.X, self.H, t)

            cost_t = self.problem.get_cost(X_t, self.varname)
            if "VERBOSE" in self.params and self.params["VERBOSE"] > 3:
                print "\t\t\t\tstepsize: ", '{:.2e}'.format(t), ",cost: ", '{:2.6f}'.format(
                    cost_t)

            if cost_t < cost + self.params["c"] * t * self.GH:  # Armijo Rule
                t_opt = t
                X_opt = X_t
                cost_opt = cost_t
                break

        return t_opt, X_opt, cost_opt

    def abort_check(self):
        progress = (self.cost_old - self.cost) / (self.cost_old + 1E-12)
        if self.t is None or self.t < self.params["t_min"] or progress < self.params["delta"]:
            return True
        else:
            return False


class CG(GDexact):
    # This class realizes the Conjugate Gradient method
    def __init__(self, problem, varname, space, params):
        super(CG, self).__init__(problem, varname, space, params)

    def prepare(self, iteration):
        G_old = self.G
        H_old = self.H
        self.prepare_grad()
        if not (iteration % self.params["direction_reset_rate"]):
            # set search direction as negative gradient at initialization or when reset due
            self.H = self.space.get_H(self.G, 0.0, 0.0)
        else:
            tauG = self.space.transport(G_old, self.X_old, H_old, self.t)
            tauH = self.space.transport(H_old, self.X_old, H_old, self.t)

            gamma_HS = innerprod(self.G - tauG, self.G) / (innerprod(self.G - tauG, tauH) + 1E-16)  # Hestenes-Stiefel update
            gamma_HS = np.maximum(gamma_HS, 0)
            # gamma_DY = innerprod(self.G, self.G) / (innerprod(self.G - tauG, tauH)) # Dai-Yuan update
            # gamma_FR = innerprod(self.G, self.G) / innerprod(tauG, tauG) # Fletcher-Reeves update
            # gamma_PR = np.maximum(innerprod(self.G, self.G - tauG) / innerprod(tauG, tauG), 0.0) # Polak-Ribiere update
            self.H = self.space.get_H(self.G, gamma_HS, tauH)

        self.GH = innerprod(self.G, self.H)
        if self.GH > 0:
            self.H = self.space.get_H(self.G, 0.0, 0.0)
            self.GH = innerprod(self.G, self.H)