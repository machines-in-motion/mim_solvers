"""__init__
License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import collections

import crocoddyl
import numpy as np

LINE_WIDTH = 100


class SQP_CPP(crocoddyl.SolverFDDP):
    def __init__(
        self, shootingProblem, use_filter_line_search=True, with_callbacks=False
    ):
        crocoddyl.SolverFDDP.__init__(self, shootingProblem)
        self.mu = 1.0
        self.termination_tolerance = 1e-6
        self.with_callbacks = with_callbacks
        self.use_filter_line_search = use_filter_line_search
        self.filter_size = 1

        self.allocateData()

    def calc(self):
        self.calcDiff()
        self.gap_norm = sum(np.linalg.norm(self.fs, 1, axis=1))
        self.merit = self.cost + self.mu * self.gap_norm
        # print(self.gap_norm)

    def computeDirection(self, kkt_check=True):
        # print("using Python")
        self.calc()
        if kkt_check:
            self.KKT_check()
            if self.KKT < self.termination_tolerance:
                return True

        self.backwardPass()
        self.computeUpdates()
        return False

    def computeUpdates(self):
        """computes step updates dx and du"""
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            # here we compute the direction
            self.lag_mul[t] = self.Vxx[t] @ (self.dx[t] - self.fs[t]) + self.Vx[t]
            self.du[t][:] = -self.K[t].dot(self.dx[t]) - self.k[t]
            A = data.Fx
            B = data.Fu
            if len(data.Fu.shape) == 1:
                bl = -B.dot(self.k[t][0])
                BL = -B.reshape(B.shape[0], 1) @ self.K[t].reshape(1, B.shape[0])
            else:
                bl = -B @ self.k[t]
                BL = -B @ self.K[t]
            self.dx[t + 1] = (A + BL) @ self.dx[t] + bl + self.fs[t + 1]

        self.lag_mul[-1] = self.Vxx[-1] @ (self.dx[-1] - self.fs[-1]) + self.Vx[-1]
        self.x_grad_norm = sum(np.linalg.norm(self.dx, 1, axis=1)) / self.problem.T
        self.u_grad_norm = sum(np.linalg.norm(self.du, 1, axis=1)) / self.problem.T
        # print("x_norm", self.x_grad_norm, "u_norm", self.u_grad_norm )

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
        self.KKT = max(self.KKT, max(abs(np.array(self.fs).flatten())))

    def tryStep(self, alpha):
        # print("using python")
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

        self.gap_norm_try = (
            sum(np.linalg.norm(self.gap_try, 1, axis=1)) / self.problem.T
        )

        self.merit_try = self.cost_try + self.mu * self.gap_norm_try
        # print("cost_try", self.cost_try, "gaps_try", self.gap_norm_try, "merit try", self.merit_try)

    def acceptStep(self):
        self.setCandidate(self.xs_try, self.us_try, False)

    def solve(
        self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None
    ):
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()]
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]

        init_xs[0] = self.problem.x0.copy()  # Initial condition guess must be x0

        self.setCandidate(init_xs, init_us, False)

        alpha = None
        self.cost_list = collections.deque(
            [np.inf] * self.filter_size, maxlen=self.filter_size
        )
        self.gap_list = collections.deque(
            [np.inf] * self.filter_size, maxlen=self.filter_size
        )

        if self.with_callbacks:
            headings = ["iter", "merit", "cost", "grad", "step", "||gaps||", "KKT"]
            print("{:>3} {:>9} {:>10} {:>11} {:>8} {:>11} {:>8}".format(*headings))

        for iter in range(maxiter):
            self.computeDirection()
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

            if self.KKT < self.termination_tolerance:
                return True

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

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod

    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]
        #
        self.dx = [np.zeros(m.state.ndx) for m in self.models()]
        self.du = [np.zeros(m.nu) for m in self.problem.runningModels]

        self.lag_mul = [np.zeros(m.state.ndx) for m in self.models()]

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
        self.expected_decrease = 0
