## Implementation of the sequential constrained lqr
## Author : Avadesh Meduri and Armand Jordana
## Date : 9/3/2022

import collections

import numpy as np
import scipy.linalg as scl
from qp_solvers.qpsolvers import QPSolvers
from qp_solvers.stagewise_qp import StagewiseADMM


def pp(s):
    return np.format_float_scientific(s, exp_digits=2, precision=4)


def rev_enumerate(l):  # noqa: E741
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class CSQP(StagewiseADMM, QPSolvers):
    def __init__(
        self,
        shootingProblem,
        method,
        use_filter_line_search=True,
        with_callbacks=False,
        qp_with_callbacks=False,
    ):
        if method == "StagewiseQP":
            StagewiseADMM.__init__(self, shootingProblem, verboseQP=qp_with_callbacks)
            self.using_qp = 0
        else:
            QPSolvers.__init__(
                self, shootingProblem, method, verboseQP=qp_with_callbacks
            )
            self.using_qp = 1

        self.mu1 = 1e1
        self.mu2 = 1e1
        self.lag_mul_inf_norm_coef = 1.01
        self.lag_mul_inf_norm = 0
        self.termination_tolerance = 1e-6

        self.iter = 0
        self.use_filter_line_search = use_filter_line_search
        self.filter_size = 1
        self.with_callbacks = with_callbacks
        self.extra_iteration_for_last_kkt = False

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod

    def tryStep(self, alpha):
        """
        This function tries the step
        """

        self.merit_try = 0
        self.constraint_norm_try = 0
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

            self.constraint_norm_try += np.linalg.norm(
                np.clip(model.g_lb - data.g, 0, np.inf), 1
            )
            self.constraint_norm_try += np.linalg.norm(
                np.clip(data.g - model.g_ub, 0, np.inf), 1
            )

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        self.constraint_norm_try += np.linalg.norm(
            np.clip(
                self.problem.terminalModel.g_lb - self.problem.terminalData.g, 0, np.inf
            ),
            1,
        )
        self.constraint_norm_try += np.linalg.norm(
            np.clip(
                self.problem.terminalData.g - self.problem.terminalModel.g_ub, 0, np.inf
            ),
            1,
        )

        self.cost_try += self.problem.terminalData.cost

        self.gap_norm_try = sum(np.linalg.norm(self.gap_try, 1, axis=1))

        if self.mu1 < 0 or self.mu2 < 0:
            self.merit_try = (
                self.cost_try
                + self.lag_mul_inf_norm_coef
                * self.lag_mul_inf_norm
                * (self.gap_norm_try + self.constraint_norm_try)
            )
        else:
            self.merit_try = (
                self.cost_try
                + self.mu1 * self.gap_norm_try
                + self.mu2 * self.constraint_norm_try
            )

    def LQ_problem_KKT_check(self):
        KKT = 0
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            Cx, Cu = data.Gx, data.Gu
            if t == 0:
                lu = (
                    data.Luu @ self.du[t]
                    + data.Lxu.T @ self.dx[t]
                    + data.Lu
                    + data.Fu.T @ self.lag_mul[t + 1]
                    + Cu.T @ self.y[t]
                )
                KKT = max(KKT, max(abs(lu)))
                continue

            lx = (
                data.Lxx @ self.dx[t]
                + data.Lxu @ self.du[t]
                + data.Lx
                + data.Fx.T @ self.lag_mul[t + 1]
                - self.lag_mul[t]
                + Cx.T @ self.y[t]
            )
            lu = (
                data.Luu @ self.du[t]
                + data.Lxu.T @ self.dx[t]
                + data.Lu
                + data.Fu.T @ self.lag_mul[t + 1]
                + Cu.T @ self.y[t]
            )
            KKT = max(KKT, max(abs(lx)), max(abs(lu)))

            l1 = np.max(
                np.abs(
                    np.clip(
                        model.g_lb - Cx @ self.dx[t] - Cu @ self.du[t] - data.g,
                        0,
                        np.inf,
                    )
                )
            )
            l2 = np.max(
                np.abs(
                    np.clip(
                        Cx @ self.dx[t] + Cu @ self.du[t] + data.g - model.g_ub,
                        0,
                        np.inf,
                    )
                )
            )
            l3 = np.max(
                np.abs(
                    self.dx[t + 1]
                    - data.Fx @ self.dx[t]
                    - data.Fu @ self.du[t]
                    - self.gap[t]
                )
            )
            KKT = max(KKT, l1, l2, l3)

        model = self.problem.terminalModel
        data = self.problem.terminalData
        Cx = data.Gx
        if model.ng != 0:
            l1 = np.max(
                np.abs(np.clip(model.g_lb - Cx @ self.dx[-1] - data.g, 0, np.inf))
            )
            l2 = np.max(
                np.abs(np.clip(Cx @ self.dx[-1] + data.g - model.g_ub, 0, np.inf))
            )
        lx = (
            self.problem.terminalData.Lxx @ self.dx[-1]
            + self.problem.terminalData.Lx
            - self.lag_mul[-1]
            + Cx.T @ self.y[-1]
        )
        KKT = max(KKT, l1, l2)
        KKT = max(KKT, max(abs(lx)))

        # Note that for this test to pass, the tolerance of the QP should be low.
        # assert KKT < 1e-6
        print("\n This should match the tolerance of the QP solver ", KKT)

    def KKT_check(self):
        if not self.using_qp:
            for t, (model, data) in enumerate(
                zip(self.problem.runningModels, self.problem.runningDatas)
            ):
                self.lag_mul[t] = self.Vxx[t] @ self.dx_tilde[t] + self.Vx[t]
            self.lag_mul[-1] = self.Vxx[-1] @ self.dx_tilde[-1] + self.Vx[-1]
        self.KKT = 0
        for t, data in enumerate(self.problem.runningDatas):
            Cx, Cu = data.Gx, data.Gu
            if t == 0:
                lu = data.Lu + data.Fu.T @ self.lag_mul[t + 1] + Cu.T @ self.y[t]
                self.KKT = max(self.KKT, max(abs(lu)))
                continue
            Cx, Cu = data.Gx, data.Gu
            lx = (
                data.Lx
                + data.Fx.T @ self.lag_mul[t + 1]
                - self.lag_mul[t]
                + Cx.T @ self.y[t]
            )
            lu = data.Lu + data.Fu.T @ self.lag_mul[t + 1] + Cu.T @ self.y[t]
            self.KKT = max(self.KKT, max(abs(lx)), max(abs(lu)))

        Cx = self.problem.terminalData.Gx
        lx = self.problem.terminalData.Lx - self.lag_mul[-1] + Cx.T @ self.y[-1]
        self.KKT = max(self.KKT, max(abs(lx)))
        self.KKT = max(self.KKT, max(abs(np.array(self.gap).flatten())))
        self.KKT = max(self.KKT, self.constraint_norm)

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
        self.constraint_list = collections.deque(
            self.filter_size * [np.inf], maxlen=self.filter_size
        )

        self.setCandidate(init_xs, init_us, False)

        alpha = None
        if self.with_callbacks:
            headings = [
                "iter",
                "merit",
                "cost",
                "||gaps||",
                "||Constraint||",
                "||(dx,du)||",
                "step",
                "KKT",
                "QP Iters",
            ]

            print(
                "{:>3} {:>7} {:>8} {:>8} {:>8} {:>11} {:>11} {:>8} {:>8}".format(
                    *headings
                )
            )
        for iter in range(maxiter):
            # self.iter = iter
            self.calc(True)
            if self.using_qp:
                self.computeDirectionFullQP()
            else:
                if iter == 0 and not self.reset_rho:
                    self.reset_rho_vec()

                self.computeDirection()
            # self.LQ_problem_KKT_check()

            self.KKT_check()
            if self.KKT < self.termination_tolerance:
                if self.with_callbacks:
                    print(
                        "{:>4} {:.4e} {:.4e} {:.4e} {:>4} {:.4e} {:>4} {:.4e} {:>4}".format(
                            "END",
                            float(self.merit),
                            self.cost,
                            self.gap_norm,
                            self.constraint_norm,
                            self.x_grad_norm + self.u_grad_norm,
                            " ---- ",
                            self.KKT,
                            self.qp_iters,
                        )
                    )
                return True

            self.gap_list.append(self.gap_norm)
            self.cost_list.append(self.cost)
            self.constraint_list.append(self.constraint_norm)
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
                        self.acceptStep(alpha)
                        break
                else:
                    if k == max_search - 1:
                        print("No improvement")
                        return False
                    if self.merit > self.merit_try:
                        self.acceptStep(alpha)
                        break

                alpha *= 0.5

            if self.with_callbacks:
                print(
                    "{:>4} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4f} {:.4e} {:>4}".format(
                        iter + 1,
                        float(self.merit),
                        self.cost,
                        self.gap_norm,
                        self.constraint_norm,
                        self.x_grad_norm + self.u_grad_norm,
                        alpha,
                        self.KKT,
                        self.qp_iters,
                    )
                )

        if self.extra_iteration_for_last_kkt:
            self.calc(True)
            if self.using_qp:
                self.computeDirectionFullQP()
            else:
                self.computeDirection()
            # if(self.BENCHMARK):
            #     self.check_qp_convergence()
            self.KKT_check()
            if self.with_callbacks:
                print(
                    "{:>4} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:>4} {:.4e} {:>4}".format(
                        "END",
                        float(self.merit),
                        self.cost,
                        self.gap_norm,
                        self.constraint_norm,
                        self.x_grad_norm + self.u_grad_norm,
                        " ---- ",
                        self.KKT,
                        " ---- ",
                    )
                )
            if self.KKT < self.termination_tolerance:
                return True

        return False
