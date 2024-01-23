import pathlib
import os
python_path = pathlib.Path('.').absolute().parent
os.sys.path.insert(1, str(python_path))

import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
from csqp import CSQP

import mim_solvers

LINE_WIDTH = 100




class DiffActionModelCLQR(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isInitial=False, isTerminal=False):
        self.nq = 2
        self.nv = 2
        self.ndx = 2
        self.nx = self.nq + self.nv
        nu = 2
        nr = 1
        ng = 4
        nh = 0
        self.isTerminal = isTerminal
        self.isInitial = isInitial

        if self.isInitial:
            ng = 0

        state = crocoddyl.StateVector(self.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, nr, ng, nh)

        if not self.isInitial:
            lower_bound = np.array([-np.inf] * ng)
            upper_bound = np.array([0.4, 0.2, np.inf, np.inf])

            self.g_lb = lower_bound
            self.g_ub = upper_bound

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

        if not self.isInitial:
            data.g = x.copy()

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

        if not self.isInitial:
            data.Gx = np.eye(self.nx)
            data.Gu = np.zeros((self.nx, self.nu))




diff_clqr_initial = DiffActionModelCLQR(isInitial=True)
diff_clqr_running = DiffActionModelCLQR()
diff_clqr_terminal = DiffActionModelCLQR(isTerminal=True)

dt = 0.1
clqr_initial = crocoddyl.IntegratedActionModelEuler(diff_clqr_initial, dt)
clqr_running = crocoddyl.IntegratedActionModelEuler(diff_clqr_running, dt)
clqr_terminal = crocoddyl.IntegratedActionModelEuler(diff_clqr_terminal, 0.0)

horizon = 100
x0 = np.zeros(4)
problem = crocoddyl.ShootingProblem(
    x0, [clqr_initial] + [clqr_running] * (horizon - 1), clqr_terminal
)

nx = 4
nu = 2

xs_init_1 = [x0] + [10 * np.ones(4)] * (horizon)
us_init_1 = [np.ones(2) for t in range(horizon)]

xs_init_2 = [x0] + [10 * np.ones(4)] * (horizon)
us_init_2 = [np.ones(2)  for t in range(horizon)]

xs_init_3 = [x0] + [10 * np.ones(4)] * (horizon)
us_init_3 = [np.ones(2)  for t in range(horizon)]

print("TEST LQ PROBLEM : CSSQP = StagewiseQP".center(LINE_WIDTH, "-"))

ddp1 = mim_solvers.SolverCSQP(problem)
ddp2 = CSQP(problem, "StagewiseQP")

ddp1.with_callbacks = True
ddp2.with_callbacks = True


ddp1.termination_tolerance = 1e-2
ddp2.termination_tolerance = 1e-2

ddp1.max_qp_iters = 2000
ddp2.max_qp_iters = 2000

ddp1.eps_abs = 1e-4
ddp1.eps_rel = 0.0

ddp2.eps_abs = 1e-4
ddp2.eps_rel = 0.0

converged = ddp1.solve(xs_init_1, us_init_1, 1)
converged = ddp2.solve(xs_init_2, us_init_2, 1)


##### UNIT TEST #####################################
set_tol = 1e-6
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.dx_tilde) - np.array(ddp2.dx_tilde)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.du_tilde) - np.array(ddp2.du_tilde)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.lag_mul) - np.array(ddp2.lag_mul)) < set_tol, "Test failed"



assert np.linalg.norm(ddp1.norm_primal - np.array(ddp2.norm_primal)) < set_tol, "Test failed"
assert np.linalg.norm(ddp1.norm_dual - np.array(ddp2.norm_dual)) < set_tol, "Test failed"
assert np.linalg.norm(ddp1.norm_primal_rel - np.array(ddp2.norm_primal_rel)) < set_tol, "Test failed"
assert np.linalg.norm(ddp1.norm_dual_rel - np.array(ddp2.norm_dual_rel)) < set_tol, "Test failed"

assert np.linalg.norm(ddp1.rho_sparse - np.array(ddp2.rho_sparse)) < set_tol, "Test failed"



for t in range(len(ddp1.rho_vec)):
    assert np.linalg.norm(ddp1.rho_vec[t] - ddp2.rho_vec[t]) < set_tol, "Test failed"
    assert np.linalg.norm(ddp1.y[t] - ddp2.y[t]) < set_tol, "Test failed"
    assert np.linalg.norm(ddp1.z[t] - ddp2.z[t]) < set_tol, "Test failed"


assert ddp1.qp_iters == ddp2.qp_iters
