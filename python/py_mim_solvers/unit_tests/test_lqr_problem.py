import pathlib
import os
python_path = pathlib.Path('.').absolute().parent
os.sys.path.insert(1, str(python_path))

import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
from sqp import SQP
from sqp_cpp import SQP_CPP



import numpy as np
import crocoddyl
import mim_solvers

LINE_WIDTH = 100

class DiffActionModelLQR(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isTerminal=False):
        self.nq = 2
        self.nv = 2
        self.ndx = 2
        self.nx = self.nq + self.nv
        nu = 2
        nr = 1
        ng = 0
        nh = 0
        self.isTerminal = isTerminal


        state = crocoddyl.StateVector(self.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, nr, ng, nh)

        self.g = np.array([0.0, -9.81])

        self.x_weights_terminal = [200, 200, 10, 10]

    def _running_cost(self, x, u):
        cost = (x[0] - 1.0) ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        cost += u[0] ** 2 + u[1] ** 2
        return 0.5 * cost

    def _terminal_cost(self, x, u):
        cost = (
            self.x_weights_terminal[0] * ((x[0] - 1.0) ** 2)
            + self.x_weights_terminal[1] * (x[1] ** 2)
            + self.x_weights_terminal[2] * (x[2] ** 2)
            + self.x_weights_terminal[3] * (x[3] ** 2)
        )
        return 0.5 * cost

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xnext = np.zeros(self.state.nx)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = u + self.g


    def calcDiff(self, data, x, u=None):
        Fx = np.zeros((2, 4))
        Fu = np.eye(2)

        Lx = np.zeros([4])
        Lu = np.zeros([2])
        Lxx = np.zeros([4, 4])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([4, 2])
        if self.isTerminal:
            Lx[0] = self.x_weights_terminal[0] * (x[0] - 1)
            Lx[1] = self.x_weights_terminal[1] * x[1]
            Lx[2] = self.x_weights_terminal[2] * x[2]
            Lx[3] = self.x_weights_terminal[3] * x[3]
            Lxx[0, 0] = self.x_weights_terminal[0]
            Lxx[1, 1] = self.x_weights_terminal[1]
            Lxx[2, 2] = self.x_weights_terminal[2]
            Lxx[3, 3] = self.x_weights_terminal[3]
        else:
            Lx[0] = x[0] - 1
            Lx[1] = x[1]
            Lx[2] = x[2]
            Lx[3] = x[3]
            Lu[0] = u[0]
            Lu[1] = u[1]
            Lxx = np.eye(4)
            Luu = np.eye(2)

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


lq_diff_running = DiffActionModelLQR()
lq_diff_terminal = DiffActionModelLQR(isTerminal=True)
dt = 0.1
horizon = 100
x0 = np.zeros(4)
lq_running = crocoddyl.IntegratedActionModelEuler(lq_diff_running, dt)
lq_terminal = crocoddyl.IntegratedActionModelEuler(lq_diff_terminal, dt)

problem = crocoddyl.ShootingProblem(x0, [lq_running] * horizon, lq_terminal)


nx = 4
nu = 2


xs_init_0 = [10*np.ones(4)] * (horizon + 1)
us_init_0 = [np.ones(2)*1 for t in range(horizon)] 

xs_init_1 = [10*np.ones(4)] * (horizon + 1)
us_init_1 = [np.ones(2)*1 for t in range(horizon)] 

xs_init_2 = [10*np.ones(4)] * (horizon + 1)
us_init_2 = [np.ones(2)*1 for t in range(horizon)] 

print("TEST 1: python SQP = mim_solvers SQP".center(LINE_WIDTH, "-"))

ddp0 = SQP(problem)
ddp1 = SQP_CPP(problem)
ddp2 = mim_solvers.SolverSQP(problem)

ddp0.with_callbacks = True
ddp1.with_callbacks = True
ddp2.with_callbacks = True

ddp0.termination_tolerance = 1e-6
ddp1.termination_tolerance = 1e-6
ddp2.termination_tolerance = 1e-6



converged = ddp0.solve(xs_init_0, us_init_0, 10)
converged = ddp1.solve(xs_init_1, us_init_1, 10)
converged = ddp2.solve(xs_init_2, us_init_2, 10)

tol = 1e-4
assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp2.xs)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp2.us)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < tol, "Test failed"


assert ddp0.iter == 0
assert ddp1.iter == 0
assert ddp2.iter == 1


print("TEST PASSED".center(LINE_WIDTH, "-"))
print("\n")