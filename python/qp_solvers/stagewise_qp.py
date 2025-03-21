## This is the implementation of the constrained sparse QP Solver
## Author : Avadesh Meduri and Armand Jordana
## Date : 8/03/2023

import eigenpy
import numpy as np
import scipy.linalg as scl
from crocoddyl import SolverAbstract

LINE_WIDTH = 100


def pp(s):
    return np.format_float_scientific(s, exp_digits=2, precision=4)


def rev_enumerate(l):  # noqa: E741
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class StagewiseADMM(SolverAbstract):
    def __init__(self, shootingProblem, verboseQP=False):
        SolverAbstract.__init__(self, shootingProblem)

        self.allocateQPData()
        self.allocateData()

        self.max_qp_iters = 1000

        self.eps_abs = 1e-4
        self.eps_rel = 1e-4
        self.adaptive_rho_tolerance = 5
        self.rho_update_interval = 25
        self.regMin = 1e-6
        self.alpha = 1.6

        self.equality_qp_initial_guess = True

        self.verboseQP = verboseQP

        self.OSQP_update = True

        self.reset_rho = False
        self.reset_y = False

        self.sigma_sparse = 1e-6
        self.rho_sparse_base = 1e-1
        self.rho_sparse = self.rho_sparse_base
        self.rho_min = 1e-6
        self.rho_max = 1e3

        if self.verboseQP:
            print("USING StagewiseQP")

    def reset_params(self):
        if self.reset_rho:
            self.reset_rho_vec()

        self.z = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]
        self.dz_relaxed = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]

        if self.reset_y:
            self.y = [np.zeros(m.ng) for m in self.problem.runningModels] + [
                np.zeros(self.problem.terminalModel.ng)
            ]

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod

    def calc(self, recalc=True):
        # compute cost and derivatives at deterministic nonlinear trajectory
        if recalc:
            self.problem.calc(self.xs, self.us)
            self.problem.calcDiff(self.xs, self.us)
        self.cost = 0
        self.constraint_norm = 0

        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            self.constraint_norm += np.linalg.norm(
                np.clip(model.g_lb - data.g, 0, np.inf), 1
            )
            self.constraint_norm += np.linalg.norm(
                np.clip(data.g - model.g_ub, 0, np.inf), 1
            )

        self.constraint_norm += np.linalg.norm(
            np.clip(
                self.problem.terminalModel.g_lb - self.problem.terminalData.g, 0, np.inf
            ),
            1,
        )
        self.constraint_norm += np.linalg.norm(
            np.clip(
                self.problem.terminalData.g - self.problem.terminalModel.g_ub, 0, np.inf
            ),
            1,
        )

        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            self.gap[t] = model.state.diff(self.xs[t + 1].copy(), data.xnext.copy())
            self.cost += data.cost
        self.cost += self.problem.terminalData.cost

        self.gap_norm = sum(np.linalg.norm(self.gap.copy(), 1, axis=1))
        self.gap = self.gap.copy()

        if self.mu1 < 0 or self.mu2 < 0:
            self.lag_mul_inf_norm = 0
            for lag_mul in self.lag_mul:
                self.lag_mul_inf_norm = max(
                    self.lag_mul_inf_norm, np.linalg.norm(lag_mul, np.inf)
                )
            for y in self.y:
                if len(y) > 0:
                    self.lag_mul_inf_norm = max(
                        self.lag_mul_inf_norm, np.linalg.norm(y, np.inf)
                    )
            self.merit = (
                self.cost
                + self.lag_mul_inf_norm_coef
                * self.lag_mul_inf_norm
                * (self.gap_norm + self.constraint_norm)
            )
        else:
            self.merit = (
                self.cost + self.mu1 * self.gap_norm + self.mu2 * self.constraint_norm
            )

    def computeDirection(self):
        self.reset_params()

        if self.equality_qp_initial_guess:
            self.backwardPass_without_constraints()
            self.forwardPass_without_constraints()

        iter = 0
        for iter in range(1, self.max_qp_iters + 1):
            if (iter) % self.rho_update_interval == 1 or iter == 1:
                self.backwardPass()

            else:
                self.backwardPass()
                # self.backwardPass_without_rho_update() # To implement

            self.forwardPass()
            self.update_lagrangian_parameters(iter)
            self.update_rho_sparse(iter)

            if (iter) % self.rho_update_interval == 0 and iter > 1:
                if (
                    self.norm_primal
                    <= self.eps_abs + self.eps_rel * self.norm_primal_rel
                    and self.norm_dual
                    <= self.eps_abs + self.eps_rel * self.norm_dual_rel
                ):
                    if self.verboseQP:
                        print(
                            "Iters",
                            iter,
                            "res-primal",
                            pp(self.norm_primal),
                            "res-dual",
                            pp(self.norm_dual),
                            "optimal rho estimate",
                            pp(self.rho_estimate_sparse),
                            "rho",
                            pp(self.rho_sparse),
                        )
                        print("StagewiseQP converged", "\n")
                    break

            if (iter) % self.rho_update_interval == 0 and iter > 1:
                if self.verboseQP:
                    print(
                        "Iters",
                        iter,
                        "res-primal",
                        pp(self.norm_primal),
                        "res-dual",
                        pp(self.norm_dual),
                        "optimal rho estimate",
                        pp(self.rho_estimate_sparse),
                        "rho",
                        pp(self.rho_sparse),
                    )
        if self.verboseQP:
            print("\n")

        self.qp_iters = iter

    def update_rho_sparse(self, iter):
        if self.OSQP_update:
            scale = (self.norm_primal * self.norm_dual_rel) / (
                self.norm_dual * self.norm_primal_rel
            )
        else:
            scale = (self.kkt_primal) / (self.kkt_dual)
        scale = np.sqrt(scale)
        self.rho_estimate_sparse = scale * self.rho_sparse
        self.rho_estimate_sparse = min(
            max(self.rho_estimate_sparse, self.rho_min), self.rho_max
        )

        if (iter) % self.rho_update_interval == 0 and iter > 1:
            if (
                self.rho_estimate_sparse > self.rho_sparse * self.adaptive_rho_tolerance
                or self.rho_estimate_sparse
                < self.rho_sparse / self.adaptive_rho_tolerance
            ):
                self.rho_sparse = self.rho_estimate_sparse
                self.apply_rho_update(self.rho_sparse)

    def apply_rho_update(self, rho_sparse):
        for t, model in enumerate(self.problem.runningModels):
            for k in range(model.ng):
                if model.g_lb[k] == -np.inf and model.g_ub[k] == np.inf:
                    self.rho_vec[t][k] = self.rho_min
                elif abs(model.g_lb[k] - model.g_ub[k]) < 1e-6:
                    self.rho_vec[t][k] = 1e3 * rho_sparse
                elif model.g_lb[k] != model.g_ub[k]:
                    self.rho_vec[t][k] = rho_sparse

        for k in range(self.problem.terminalModel.ng):
            if (
                self.problem.terminalModel.g_lb[k] == -np.inf
                and self.problem.terminalModel.g_ub[k] == np.inf
            ):
                self.rho_vec[-1][k] = self.rho_min
            elif (
                abs(
                    self.problem.terminalModel.g_lb[k]
                    - self.problem.terminalModel.g_ub[k]
                )
                < 1e-6
            ):
                self.rho_vec[-1][k] = 1e3 * rho_sparse
            elif (
                self.problem.terminalModel.g_lb[k] != self.problem.terminalModel.g_ub[k]
            ):
                self.rho_vec[-1][k] = rho_sparse

    def update_lagrangian_parameters(self, iter):
        self.norm_primal = -np.inf
        self.norm_dual = -np.inf
        self.kkt_primal = -np.inf
        self.kkt_dual = -np.inf

        self.norm_primal_rel, self.norm_dual_rel = [-np.inf, -np.inf], -np.inf

        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            if model.ng == 0:
                self.dx[t] = self.dx_tilde[t].copy()
                self.du[t] = self.du_tilde[t].copy()
                continue

            Cx, Cu = data.Gx, data.Gu

            z_prev = self.z[t].copy()

            Cdx_Cdu = Cx @ self.dx_tilde[t].copy() + Cu @ self.du_tilde[t].copy()

            self.dz_relaxed[t] = self.alpha * (Cdx_Cdu) + (1 - self.alpha) * self.z[t]

            self.z[t] = np.clip(
                self.dz_relaxed[t] + np.divide(self.y[t], self.rho_vec[t]),
                model.g_lb - data.g,
                model.g_ub - data.g,
            )
            self.y[t] += np.multiply(self.rho_vec[t], (self.dz_relaxed[t] - self.z[t]))

            self.dx[t] = self.dx_tilde[t].copy()
            self.du[t] = self.du_tilde[t].copy()

            # OSQP
            if (iter) % self.rho_update_interval == 0 and iter > 1:
                dual_vecx = Cx.T @ np.multiply(self.rho_vec[t], (self.z[t] - z_prev))
                dual_vecu = Cu.T @ np.multiply(self.rho_vec[t], (self.z[t] - z_prev))
                self.norm_dual = max(
                    self.norm_dual, max(abs(dual_vecx)), max(abs(dual_vecu))
                )
                self.norm_primal = max(self.norm_primal, max(abs(Cdx_Cdu - self.z[t])))

                # KKT
                dual_vecx = (
                    data.Lxx @ self.dx[t]
                    + data.Lxu @ self.du[t]
                    + data.Lx
                    + data.Fx.T @ self.lag_mul[t + 1]
                    - self.lag_mul[t]
                    + Cx.T @ self.y[t]
                )
                dual_vecu = (
                    data.Luu @ self.du[t]
                    + data.Lxu.T @ self.dx[t]
                    + data.Lu
                    + data.Fu.T @ self.lag_mul[t + 1]
                    + Cu.T @ self.y[t]
                )
                self.kkt_dual = max(
                    self.kkt_dual, max(abs(dual_vecx)), max(abs(dual_vecu))
                )
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
                self.kkt_primal = max(self.kkt_primal, l1, l2)

                self.norm_primal_rel[0] = max(
                    self.norm_primal_rel[0], max(abs(Cdx_Cdu))
                )
                self.norm_primal_rel[1] = max(
                    self.norm_primal_rel[1], max(abs(self.z[t]))
                )
                self.norm_dual_rel = max(
                    self.norm_dual_rel,
                    max(abs(Cx.T @ self.y[t])),
                    max(abs(Cu.T @ self.y[t])),
                )

        self.dx[-1] = self.dx_tilde[-1].copy()
        if self.problem.terminalModel.ng != 0:
            model = self.problem.terminalModel
            data = self.problem.terminalData
            Cx = data.Gx

            z_prev = self.z[-1].copy()

            Cdx = Cx @ self.dx_tilde[-1].copy()

            self.dz_relaxed[-1] = self.alpha * Cdx + (1 - self.alpha) * self.z[-1]
            self.z[-1] = np.clip(
                self.dz_relaxed[-1] + np.divide(self.y[-1], self.rho_vec[-1]),
                model.g_lb - data.g,
                model.g_ub - data.g,
            )
            # print(data.g)
            # print(self.dz_relaxed[-1])

            self.y[-1] += np.multiply(
                self.rho_vec[-1], (self.dz_relaxed[-1] - self.z[-1])
            )

            self.dx[-1] = self.dx_tilde[-1].copy()

            # OSQP
            if (iter) % self.rho_update_interval == 0 and iter > 1:
                self.norm_primal = max(self.norm_primal, max(abs(Cdx - self.z[-1])))
                dual_vec = Cx.T @ np.multiply(self.rho_vec[-1], (self.z[-1] - z_prev))
                self.norm_dual = max(self.norm_dual, max(abs(dual_vec)))

                # KKT
                dual_vec = (
                    self.problem.terminalData.Lxx @ self.dx[-1]
                    + self.problem.terminalData.Lx
                    - self.lag_mul[-1]
                    + Cx.T @ self.y[-1]
                )
                self.kkt_dual = max(self.kkt_dual, max(abs(dual_vec)))
                l1 = np.max(
                    np.abs(np.clip(model.g_lb - Cx @ self.dx[-1] - data.g, 0, np.inf))
                )
                l2 = np.max(
                    np.abs(np.clip(Cx @ self.dx[-1] + data.g - model.g_ub, 0, np.inf))
                )
                self.kkt_primal = max(self.kkt_primal, l1, l2)

                self.norm_primal_rel[0] = max(
                    self.norm_primal_rel[0], max(abs(Cx @ self.dx[-1]))
                )
                self.norm_primal_rel[1] = max(
                    self.norm_primal_rel[1], max(abs(self.z[-1]))
                )
                self.norm_dual_rel = max(
                    self.norm_dual_rel, max(abs(Cx.T @ self.y[-1]))
                )
        self.norm_primal_rel = max(self.norm_primal_rel)

    def acceptStep(self, alpha):
        for t, (model, _) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha * self.dx[t])
            self.us_try[t] = self.us[t] + alpha * self.du[t]
        self.xs_try[-1] = model.state.integrate(
            self.xs[-1], alpha * self.dx[-1]
        )  ## terminal state update

        self.setCandidate(self.xs_try, self.us_try, False)

    def forwardPass(self):
        """computes step updates dx and du"""
        assert np.linalg.norm(self.dx[0]) < 1e-6
        for t, (_, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            # self.lag_mul[t] = self.Vxx[t] @ self.dx_tilde[t] + self.Vx[t]
            self.du_tilde[t][:] = -self.K[t].dot(self.dx_tilde[t]) - self.k[t]
            A = data.Fx.copy()
            B = data.Fu.copy()
            if len(B.shape) == 1:
                bk = B.dot(self.k[t][0])
                BK = B.reshape(B.shape[0], 1) @ self.K[t]
            else:
                bk = B @ self.k[t]
                BK = B @ self.K[t]
            self.dx_tilde[t + 1] = (A - BK) @ self.dx_tilde[t] - bk + self.gap[t].copy()

        # self.lag_mul[-1] = self.Vxx[-1] @ self.dx_tilde[-1] + self.Vx[-1]

        self.x_grad_norm = np.linalg.norm(self.dx_tilde) / (self.problem.T + 1)
        self.u_grad_norm = np.linalg.norm(self.du_tilde) / self.problem.T

    def forwardPass_without_constraints(self):
        """computes step updates dx and du"""
        assert np.linalg.norm(self.dx[0]) < 1e-6
        for t, data in enumerate(self.problem.runningDatas):
            self.du[t][:] = -self.K[t].dot(self.dx[t]) - self.k[t]
            A = data.Fx.copy()
            B = data.Fu.copy()
            if len(B.shape) == 1:
                bk = B.dot(self.k[t][0])
                BK = B.reshape(B.shape[0], 1) @ self.K[t]
            else:
                bk = B @ self.k[t]
                BK = B @ self.K[t]
            self.dx[t + 1] = (A - BK) @ self.dx[t] - bk + self.gap[t].copy()

    def backwardPass(self):
        rho_mat = self.rho_vec[-1] * np.eye(len(self.rho_vec[-1]))
        self.Vxx[-1][:, :] = (
            self.problem.terminalData.Lxx.copy()
            + self.sigma_sparse * np.eye(self.problem.terminalModel.state.ndx)
        )
        self.Vx[-1][:] = (
            self.problem.terminalData.Lx.copy() - self.sigma_sparse * self.dx[-1]
        )
        if self.problem.terminalModel.ng != 0:
            Cx = self.problem.terminalData.Gx
            self.Vxx[-1][:, :] += Cx.T @ rho_mat @ Cx
            self.Vx[-1][:] += Cx.T @ (self.y[-1] - rho_mat @ self.z[-1])[:]

        for t, (model, data) in rev_enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            A = data.Fx.copy()
            B = data.Fu.copy()

            rho_mat = self.rho_vec[t] * np.eye(len(self.rho_vec[t]))
            Vx_p = self.Vx[t + 1] + self.Vxx[t + 1] @ self.gap[t].copy()

            Qx = data.Lx.copy() + A.T @ Vx_p
            Qxx = data.Lxx.copy() + A.T @ self.Vxx[t + 1] @ A
            Qu = data.Lu.copy() + B.T @ Vx_p
            Quu = data.Luu.copy() + B.T @ self.Vxx[t + 1] @ B
            Qxu = data.Lxu.copy() + A.T @ self.Vxx[t + 1] @ B

            if model.ng != 0:
                Cx, Cu = data.Gx, data.Gu
                if t > 0:
                    Qx += Cx.T @ (self.y[t] - rho_mat @ self.z[t])[:]
                    Qxx += Cx.T @ rho_mat @ Cx
                    Qxu += Cx.T @ rho_mat @ Cu
                Qu += Cu.T @ (self.y[t] - rho_mat @ self.z[t])[:]
                Quu += Cu.T @ rho_mat @ Cu

            Qxx += self.sigma_sparse * np.eye(model.state.ndx)
            Quu += self.sigma_sparse * np.eye(model.nu)
            Qx -= self.sigma_sparse * self.dx[t]
            Qu -= self.sigma_sparse * self.du[t]

            Q_llt = eigenpy.LLT(Quu)
            self.K[t][:, :] = Q_llt.solve(Qxu.T)
            self.k[t][:] = Q_llt.solve(Qu)

            # import pdb; pdb.set_trace()
            self.Vx[t] = Qx - self.K[t].T @ Qu
            self.Vxx[t] = Qxx - Qxu @ self.K[t]

            Vxx_tmp_ = 0.5 * (self.Vxx[t] + self.Vxx[t].T)
            self.Vxx[t] = Vxx_tmp_

    def backwardPass_without_constraints(self):
        self.Vxx[-1][:, :] = self.problem.terminalData.Lxx.copy()
        self.Vx[-1][:] = self.problem.terminalData.Lx.copy()

        for t, (model, data) in rev_enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            A = data.Fx.copy()
            B = data.Fu.copy()

            Vx_p = self.Vx[t + 1] + self.Vxx[t + 1] @ self.gap[t].copy()
            Qx = data.Lx.copy() + A.T @ Vx_p
            Qxx = data.Lxx.copy() + A.T @ self.Vxx[t + 1] @ A
            Qu = data.Lu.copy() + B.T @ Vx_p
            Quu = data.Luu.copy() + B.T @ self.Vxx[t + 1] @ B
            Qxu = data.Lxu.copy() + A.T @ self.Vxx[t + 1] @ B

            Q_llt = eigenpy.LLT(Quu)
            self.K[t][:, :] = Q_llt.solve(Qxu.T)
            self.k[t][:] = Q_llt.solve(Qu)

            # import pdb; pdb.set_trace()
            self.Vx[t] = Qx - self.K[t].T @ Qu
            self.Vxx[t] = Qxx - Qxu @ self.K[t]

            Vxx_tmp_ = 0.5 * (self.Vxx[t] + self.Vxx[t].T)
            self.Vxx[t] = Vxx_tmp_

    def solve(
        self, init_xs=None, init_us=None, maxiter=1000, isFeasible=False, regInit=None
    ):
        # ___________________ Initialize ___________________#
        self.max_qp_iters = maxiter
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()]
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]

        init_xs[0][:] = self.problem.x0.copy()  # Initial condition guess must be x0
        self.setCandidate(init_xs, init_us, False)
        self.computeDirection(KKT=False)

        self.acceptStep(alpha=1.0)

    def allocateQPData(self):
        self.y = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]
        self.z = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]

        self.dx = [np.zeros(m.state.ndx) for m in self.models()]
        self.du = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.dx_tilde = [np.zeros(m.state.ndx) for m in self.models()]
        self.du_tilde = [np.zeros(m.nu) for m in self.problem.runningModels]
        #
        self.dx_test = [np.zeros(m.state.ndx) for m in self.models()]
        self.du_test = [np.zeros(m.nu) for m in self.problem.runningModels]
        #
        self.lag_mul = [np.zeros(m.state.ndx) for m in self.models()]
        self.dz_relaxed = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]

        self.rho_vec = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]

    def reset_rho_vec(self):
        self.rho_sparse = self.rho_sparse_base
        self.apply_rho_update(self.rho_sparse)

    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]
        #
        self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]
        self.Vx = [np.zeros(m.state.ndx) for m in self.models()]
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]

        self.H = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]
        self.G = [
            np.zeros([m.state.ndx, m.state.ndx]) for m in self.problem.runningModels
        ]
        self.H_llt = [np.array([0]) for m in self.problem.runningModels]
        #
        self.x_grad = [np.zeros(m.state.ndx) for m in self.models()]
        self.u_grad = [np.zeros(m.nu) for m in self.problem.runningModels]

        self.gap = [np.zeros(m.state.ndx) for m in self.models()]  # gaps
        self.gap_try = [
            np.zeros(m.state.ndx) for m in self.models()
        ]  # gaps for line search

        self.x_grad_norm = 0
        self.u_grad_norm = 0
        self.gap_norm = 0
        self.cost = 0
        self.cost_try = 0

        self.constraint_norm = 0
        self.constraint_norm_try = 0
        #

        self.ndx = self.problem.terminalModel.state.ndx
        self.nu = self.problem.runningModels[0].nu

        self.y = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]
        self.z = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]

        self.qp_iters = 0

        n_eq = (
            sum([m.nh for m in self.problem.runningModels])
            + self.problem.terminalModel.nh
        )
        assert n_eq == 0
