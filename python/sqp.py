"""__init__
License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import collections

import numpy as np
import scipy.linalg as scl
from crocoddyl import SolverAbstract

LINE_WIDTH = 100


def rev_enumerate(l):  # noqa: E741
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class SQP(SolverAbstract):
    def __init__(
        self, shootingProblem, use_filter_line_search=True, with_callbacks=False
    ):
        SolverAbstract.__init__(self, shootingProblem)
        # self.x_reg = 0
        # self.u_reg = 0
        # self.regFactor = 10
        # self.regMax = 1e9
        # self.regMin = 1e-9
        self.mu = 1.0
        self.termination_tolerance = 1e-6

        self.use_filter_line_search = use_filter_line_search
        self.filter_size = 1
        self.with_callbacks = with_callbacks

        self.extra_iteration_for_last_kkt = False
        self.allocateData()

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod

    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory
        self.problem.calc(self.xs, self.us)
        self.problem.calcDiff(self.xs, self.us)
        self.cost = 0
        # self.merit_old = self.merit

        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            # model.calc(data, self.xs[t], self.us[t])
            self.gap[t] = model.state.diff(self.xs[t + 1], data.xnext)  # gaps
            self.cost += data.cost

        self.gap_norm = sum(np.linalg.norm(self.gap.copy(), 1, axis=1))

        self.cost += self.problem.terminalData.cost
        self.merit = self.cost + self.mu * self.gap_norm

    def computeDirection(self, recalc=True):
        self.calc()
        self.backwardPass()
        self.computeUpdates()

        # self.compute_expected_decrease()

    def LQ_problem_KKT_check(self):
        KKT = 0
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            KKT += max(
                abs(
                    data.Lxx @ self.dx[t]
                    + data.Lxu @ self.du[t]
                    + data.Lx
                    + data.Fx.T @ self.lag_mul[t + 1]
                    - self.lag_mul[t]
                )
            )
            KKT += max(
                abs(
                    data.Luu @ self.du[t]
                    + data.Lxu.T @ self.dx[t]
                    + data.Lu
                    + data.Fu.T @ self.lag_mul[t + 1]
                )
            )

        KKT += max(
            abs(
                self.problem.terminalData.Lxx @ self.dx[-1]
                + self.problem.terminalData.Lx
                - self.lag_mul[-1]
            )
        )
        # print("\n THIS SHOULD BE ZERO ", KKT)

        # print("\n")

    def KKT_check(self):
        self.KKT = 0
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            self.KKT = max(
                self.KKT,
                max(abs(data.Lx + data.Fx.T @ self.lag_mul[t + 1] - self.lag_mul[t])),
            )
            self.KKT = max(
                self.KKT, max(abs(data.Lu + data.Fu.T @ self.lag_mul[t + 1]))
            )

        self.KKT = max(
            self.KKT, max(abs(self.problem.terminalData.Lx - self.lag_mul[-1]))
        )
        self.KKT = max(self.KKT, max(abs(np.array(self.gap).flatten())))

    def computeUpdates(self):
        """computes step updates dx and du"""
        self.expected_decrease = 0
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            # here we compute the direction
            self.lag_mul[t] = self.S[t] @ self.dx[t] + self.s[t]
            self.du[t][:] = self.L[t].dot(self.dx[t]) + self.l[t]
            A = data.Fx
            B = data.Fu
            if len(data.Fu.shape) == 1:
                bl = B.dot(self.l[t][0])
                BL = B.reshape(B.shape[0], 1) @ self.L[t]
            else:
                bl = B @ self.l[t]
                BL = B @ self.L[t]
            self.dx[t + 1] = (A + BL) @ self.dx[t] + bl + self.gap[t]

        self.lag_mul[-1] = self.S[-1] @ self.dx[-1] + self.s[-1]
        self.x_grad_norm = sum(np.linalg.norm(self.dx, 1, axis=1)) / (
            self.problem.T + 1
        )
        self.u_grad_norm = sum(np.linalg.norm(self.du, 1, axis=1)) / self.problem.T

    def compute_expected_decrease(self):
        self.expected_decrease = 0
        self.rho = 0.999999
        for t, (_, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            q = data.Lx
            r = data.Lu
            self.expected_decrease += q.T @ self.dx[t] + r.T @ self.du[t]
            # hess_decrease = (
            #     self.du[t].T @ data.Luu @ self.du[t]
            #     + self.dx[t].T @ data.Lxx @ self.dx[t]
            # )

        q = self.problem.terminalData.Lx
        self.expected_decrease += q.T @ self.dx[-1]

        # hess_decrease = self.dx[-1].T @ data.Lxx @ self.dx[-1]
        tmp_mu = self.expected_decrease / ((1 - self.rho) * self.gap_norm)
        self.mu = tmp_mu

    def tryStep(self, alpha):
        """
        This function tries the step
        """

        self.merit_try = 0
        self.cost_try = 0
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha * self.dx[t])
            self.us_try[t] = self.us[t] + alpha * self.du[t]
        self.xs_try[-1] = model.state.integrate(
            self.xs[-1], alpha * self.dx[-1]
        )  ## terminal state update

        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            model.calc(data, self.xs_try[t], self.us_try[t])
            self.gap_try[t] = model.state.diff(self.xs_try[t + 1], data.xnext)  # gaps
            self.cost_try += data.cost

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        self.cost_try += self.problem.terminalData.cost

        self.gap_norm_try = sum(np.linalg.norm(self.gap_try, 1, axis=1))

        self.merit_try = self.cost_try + self.mu * self.gap_norm_try

    def acceptStep(self):
        self.setCandidate(self.xs_try, self.us_try, False)

    def backwardPass(self):
        self.S[-1][:, :] = self.problem.terminalData.Lxx
        self.s[-1][:] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            r = data.Lu
            q = data.Lx
            R = data.Luu
            Q = data.Lxx
            P = data.Lxu.T
            A = data.Fx
            B = data.Fu

            h = r + B.T @ (self.s[t + 1] + self.S[t + 1] @ self.gap[t])
            G = P + B.T @ self.S[t + 1] @ A
            self.H = R + B.T @ self.S[t + 1] @ B  # + 1e-9 * np.eye(model.nu)

            if len(G.shape) == 1:
                G = np.resize(G, (1, G.shape[0]))
            ## Making sure H is PD
            Lb_uu = scl.cho_factor(self.H, lower=True)
            # while True:
            #     try:
            #         Lb_uu = scl.cho_factor(self.H, lower=True)
            #         break
            #     except:
            #         print("increasing H")
            #         self.H += 100*self.regMin*np.eye(len(self.H))

            H = self.H.copy()
            self.L[t][:, :] = -1 * scl.cho_solve(Lb_uu, G)
            self.l[t][:] = -1 * scl.cho_solve(Lb_uu, h)

            self.S[t] = Q + A.T @ (self.S[t + 1]) @ A - self.L[t].T @ H @ self.L[t]
            self.S[t] = 0.5 * (self.S[t] + self.S[t].T)
            self.s[t] = (
                q
                + A.T @ (self.S[t + 1] @ self.gap[t] + self.s[t + 1])
                + G.T @ self.l[t][:]
                + self.L[t].T @ (h + H @ self.l[t][:])
            )

    def solve(
        self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None
    ):
        # ___________________ Initialize ___________________#
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()]
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]

        init_xs[0] = self.problem.x0.copy()  # Initial condition guess must be x0

        self.gap_list = collections.deque(
            self.filter_size * [np.inf], maxlen=self.filter_size
        )
        self.cost_list = collections.deque(
            self.filter_size * [np.inf], maxlen=self.filter_size
        )

        self.setCandidate(init_xs, init_us, False)

        alpha = None
        if self.with_callbacks:
            headings = ["iter", "merit", "cost", "grad", "step", "||gaps||", "KKT"]

            print("{:>3} {:>9} {:>10} {:>11} {:>8} {:>11} {:>8}".format(*headings))
        for iter in range(maxiter):
            recalc = True  # this will recalculated derivatives in Compute Direction
            self.computeDirection(recalc=recalc)

            # self.check_optimality()

            self.KKT_check()
            if self.KKT < self.termination_tolerance:
                if self.with_callbacks:
                    print(
                        "{:>4} {:.5e} {:.5e} {:.5e} {:.4f} {:.5e} {:.5e}".format(
                            iter,
                            float(self.merit),
                            self.cost,
                            self.x_grad_norm + self.u_grad_norm,
                            alpha,
                            self.gap_norm,
                            self.KKT,
                        )
                    )
                return True

            self.gap_list.append(self.gap_norm)
            self.cost_list.append(self.cost)
            alpha = 1.0
            max_search = 10
            for k in range(max_search):
                self.tryStep(alpha)
                if self.use_filter_line_search:
                    is_worse_than_memory = False
                    count = 0
                    while (
                        count < self.filter_size
                        and not is_worse_than_memory
                        and count <= iter
                    ):
                        is_worse_than_memory = (
                            self.cost_list[self.filter_size - 1 - count] < self.cost_try
                            and self.gap_list[self.filter_size - 1 - count]
                            < self.gap_norm_try
                        )
                        count += 1

                    if not is_worse_than_memory:
                        self.acceptStep()
                        break
                else:
                    if k == max_search - 1:
                        print("No improvement")
                        return False
                    if self.merit > self.merit_try:
                        self.acceptStep()
                        break

                alpha *= 0.5

            if self.with_callbacks:
                print(
                    "{:>4} {:.5e} {:.5e} {:.5e} {:.4f} {:.5e} {:.5e}".format(
                        iter,
                        float(self.merit),
                        self.cost,
                        self.x_grad_norm + self.u_grad_norm,
                        alpha,
                        self.gap_norm,
                        self.KKT,
                    )
                )

        if self.extra_iteration_for_last_kkt:
            recalc = True  # this will recalculated derivatives in Compute Direction
            self.computeDirection(recalc=recalc)

            # self.check_optimality()

            self.KKT_check()
            if self.KKT < self.termination_tolerance:
                if self.with_callbacks:
                    print(
                        "{:>4} {:.5e} {:.5e} {:.5e} {:.4f} {:.5e} {:.5e}".format(
                            iter,
                            float(self.merit),
                            self.cost,
                            self.x_grad_norm + self.u_grad_norm,
                            alpha,
                            self.gap_norm,
                            self.KKT,
                        )
                    )
                return True
        return False

    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]
        #
        self.dx = [np.zeros(m.state.ndx) for m in self.models()]
        self.du = [np.zeros(m.nu) for m in self.problem.runningModels]
        #
        self.lag_mul = [np.zeros(m.state.ndx) for m in self.models()]
        #
        self.S = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]
        self.s = [np.zeros(m.state.ndx) for m in self.models()]
        self.L = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.l = [np.zeros([m.nu]) for m in self.problem.runningModels]
        #
        self.x_grad = [np.zeros(m.state.ndx) for m in self.models()]
        self.u_grad = [np.zeros(m.nu) for m in self.problem.runningModels]

        self.gap = [np.zeros(m.state.ndx) for m in self.models()]  # gaps
        self.gap_try = [
            np.zeros(m.state.ndx) for m in self.models()
        ]  # gaps for line search

        self.merit = 0
        self.merit_old = 0
        self.x_grad_norm = 0
        self.u_grad_norm = 0
        self.gap_norm = 0
        self.cost = 0
        self.cost_try = 0

    def check_optimality(self):
        """
        This function checks if the convexified lqr problem reaches optimality before we take the next step of the SQP
        """
        error = 0
        for t, (_, data) in rev_enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            r = data.Lu
            R = data.Luu
            P = data.Lxu.T
            A = data.Fx
            B = data.Fu

            h = r + B.T @ (self.s[t + 1] + self.S[t + 1] @ self.gap[t])
            G = P + B.T @ self.S[t + 1] @ A
            H = R + B.T @ self.S[t + 1] @ B

            error += np.linalg.norm(
                H @ self.du[t] + h + G @ self.dx[t]
            )  ## optimality check

        assert error < 1e-6
