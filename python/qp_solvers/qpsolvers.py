## This contains various QP solvers that solve the convex subproblem of the sqp
## Author : Avadesh Meduri & Armand Jordana
## Date : 21/03/2022

import importlib.util
import time

import numpy as np
import osqp
import proxsuite
from crocoddyl import SolverAbstract
from scipy import sparse

from .py_osqp import CustomOSQP
from .stagewise_qp_kkt import StagewiseQPKKT

HPIPM_PYTHON_FOUND = importlib.util.find_spec("hpipm_python")
if HPIPM_PYTHON_FOUND:
    import hpipm_python


class QPSolvers(SolverAbstract, CustomOSQP, StagewiseQPKKT):
    def __init__(self, shootingProblem, method, verboseQP=True):
        SolverAbstract.__init__(self, shootingProblem)

        assert (
            method == "ProxQP"
            or method == "OSQP"
            or method == "CustomOSQP"
            or method == "StagewiseQPKKT"
            or method == "HPIPM_DENSE"
            or method == "HPIPM_OCP"
        )
        self.method = method
        if method == "CustomOSQP":
            CustomOSQP.__init__(self)
        if method == "StagewiseQPKKT":
            StagewiseQPKKT.__init__(self)
        self.allocateDataQP()
        self.max_qp_iters = 1000
        self.initialize = True
        self.verboseQP = verboseQP
        self.eps_abs = 1e-4
        self.eps_rel = 0.0
        self.OSQP_scaling = False

        self.BENCHMARK = True
        self.DEBUG = False

        if self.verboseQP:
            print("USING " + str(method))

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
            # Inequality
            self.constraint_norm += np.linalg.norm(
                np.clip(model.g_lb - data.g, 0, np.inf), 1
            )
            self.constraint_norm += np.linalg.norm(
                np.clip(data.g - model.g_ub, 0, np.inf), 1
            )
            # Equality
            # self.constraint_norm += np.linalg.norm(data.h - model.h_ub, 0, np.inf), 1)

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
            self.gap[t] = model.state.diff(
                self.xs[t + 1].copy(), data.xnext.copy()
            )  # gaps
            self.cost += data.cost

        self.gap_norm = sum(np.linalg.norm(self.gap.copy(), 1, axis=1))
        self.cost += self.problem.terminalData.cost
        self.gap = self.gap.copy()

    def computeDirectionFullQP(self):
        self.n_vars = self.problem.T * (self.ndx + self.nu)

        P = np.zeros(
            (
                self.problem.T * (self.ndx + self.nu),
                self.problem.T * (self.ndx + self.nu),
            )
        )
        q = np.zeros(self.problem.T * (self.ndx + self.nu))

        Asize = self.problem.T * (self.ndx + self.nu)
        A = np.zeros((self.problem.T * self.ndx, Asize))
        B = np.zeros(self.problem.T * self.ndx)

        NNZ_block_P = 0
        NNZ_block_A = 0
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            index_u = self.problem.T * self.ndx + t * self.nu
            if t >= 1:
                index_x = (t - 1) * self.ndx
                P[index_x : index_x + self.ndx, index_x : index_x + self.ndx] = (
                    data.Lxx.copy()
                )
                NNZ_block_P += np.size(data.Lxx)
                P[index_x : index_x + self.ndx, index_u : index_u + self.nu] = (
                    data.Lxu.copy()
                )
                NNZ_block_P += np.size(data.Lxu)
                P[index_u : index_u + self.nu, index_x : index_x + self.ndx] = (
                    data.Lxu.T.copy()
                )
                NNZ_block_P += np.size(data.Lxu.T)
                q[index_x : index_x + self.ndx] = data.Lx.copy()

            P[index_u : index_u + self.nu, index_u : index_u + self.nu] = (
                data.Luu.copy()
            )
            NNZ_block_P += np.size(data.Luu)
            q[index_u : index_u + self.nu] = data.Lu.copy()

            A[
                t * self.ndx : (t + 1) * self.ndx, index_u : index_u + self.nu
            ] = -data.Fu.copy()
            NNZ_block_A += np.size(data.Fu)
            A[t * self.ndx : (t + 1) * self.ndx, t * self.ndx : (t + 1) * self.ndx] = (
                np.eye(self.ndx)
            )
            NNZ_block_A += self.ndx

            if t >= 1:
                A[
                    t * self.ndx : (t + 1) * self.ndx, (t - 1) * self.ndx : t * self.ndx
                ] = -data.Fx.copy()
                NNZ_block_A += np.size(data.Fx)
            B[t * self.ndx : (t + 1) * self.ndx] = self.gap[t].copy()

        P[
            (self.problem.T - 1) * self.ndx : self.problem.T * self.ndx,
            self.problem.T * self.ndx - self.ndx : self.problem.T * self.ndx,
        ] = self.problem.terminalData.Lxx.copy()
        NNZ_block_P += np.size(self.problem.terminalData.Lxx)
        q[(self.problem.T - 1) * self.ndx : self.problem.T * self.ndx] = (
            self.problem.terminalData.Lx.copy()
        )

        n = self.problem.T * (self.ndx + self.nu)
        self.n_eq = self.problem.T * self.ndx

        self.n_in = sum([len(self.y[i]) for i in range(len(self.y))])

        C = np.zeros((self.n_in, self.problem.T * (self.ndx + self.nu)))
        l = np.zeros(self.n_in)  # noqa: E741
        u = np.zeros(self.n_in)

        nin_count = 0
        index_x = self.problem.T * self.ndx
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            if model.ng == 0:
                continue
            l[nin_count : nin_count + model.ng] = model.g_lb - data.g
            u[nin_count : nin_count + model.ng] = model.g_ub - data.g
            if t > 0:
                C[
                    nin_count : nin_count + model.ng, (t - 1) * self.ndx : t * self.ndx
                ] = data.Gx
                NNZ_block_A += np.size(data.Gx)
            C[
                nin_count : nin_count + model.ng,
                index_x + t * self.nu : index_x + (t + 1) * self.nu,
            ] = data.Gu
            NNZ_block_A += np.size(data.Gu)
            nin_count += model.ng

        model = self.problem.terminalModel
        data = self.problem.terminalData
        if model.ng != 0:
            l[nin_count : nin_count + model.ng] = model.g_lb - data.g
            u[nin_count : nin_count + model.ng] = model.g_ub - data.g
            C[
                nin_count : nin_count + model.ng,
                (self.problem.T - 1) * self.ndx : self.problem.T * self.ndx,
            ] = data.Gx
            NNZ_block_A += np.size(data.Gx)
            nin_count += model.ng

        # If we are benchmarking, setup the problem using QPSolvers (S. Caron)
        # in order to get the unified convergence metrics (prim_res, dual_res, dual_gap)
        if self.BENCHMARK:
            # We follow the convention of QPSolvers package, i.e.
            # min_x (1/2) x^T P x + q^T x
            #   s.t. Ax  = b
            #        Gx <= h
            # see here https://github.com/qpsolvers/qpsolvers/blob/main/qpsolvers/problem.py
            P_qp = P
            q_qp = q
            A_qp = A
            b_qp = B
            G_qp = np.vstack([C, -C])
            h_qp = np.hstack([u, -l])
            # prob_qp = Problem(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp) # no box constraints
            # sol_qp = Solution(prob_qp)

        if self.method == "ProxQP":
            qp = proxsuite.proxqp.sparse.QP(n, self.n_eq, self.n_in)
            qp.settings.eps_abs = self.eps_abs
            qp.settings.eps_rel = self.eps_rel
            qp.settings.max_iter = self.max_qp_iters
            qp.settings.initial_guess = proxsuite.proxqp.NO_INITIAL_GUESS
            qp.init(P, q, A, B, C, l, u)
            t1 = time.time()
            qp.solve()
            self.qp_time = time.time() - t1
            self.qp_iters = qp.results.info.iter
            self.found_qp_sol = (
                qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
            )
            # Get solution
            res = qp.results.x
            self.z_k = qp.results.z
            self.y_k = qp.results.y
            # If we are benchmarking the QP, check convergence using dual and primal residuals
            # Check here : https://scaron.info/blog/optimality-conditions-and-numerical-tolerances-in-qp-solvers.html
            if self.BENCHMARK:
                self.norm_dual = np.max(
                    np.abs(P @ res + q + A.T @ self.y_k + C.T @ self.z_k)
                )
                self.norm_primal = np.max(np.abs(A @ res - B))
                if C.shape[0] > 0:
                    res_lb = np.max(np.abs(np.clip(l - C @ res, 0, np.inf)))
                    res_ub = np.max(np.abs(np.clip(C @ res - u, 0, np.inf)))
                    self.norm_primal = np.maximum(res_lb, self.norm_primal)
                    self.norm_primal = np.maximum(res_ub, self.norm_primal)
                print(f"- Primal residual: {self.norm_primal:.1e}")
                print(f"- Dual residual: {self.norm_dual:.1e}")

            # Fill out Lagrange mutlipliers for SQP
            for t in range(self.problem.T):
                self.lag_mul[t + 1] = -qp.results.y[t * self.ndx : (t + 1) * self.ndx]
            nin_count = 0
            for t in range(self.problem.T + 1):
                if t < self.problem.T:
                    model = self.problem.runningModels[t]
                else:
                    model = self.problem.terminalModel
                if model.ng == 0:
                    continue
                self.y[t] = qp.results.z[nin_count : nin_count + model.ng]
                nin_count += model.ng

        elif self.method == "OSQP":
            Aeq = sparse.csr_matrix(A)
            Aineq = sparse.csr_matrix(C)
            Aosqp = sparse.vstack([Aeq, Aineq])
            losqp = np.hstack([B, l])
            uosqp = np.hstack([B, u])
            P = sparse.csr_matrix(P)
            if self.DEBUG:
                print(
                    "nnz(P) = ",
                    P.count_nonzero(),
                    " out of ",
                    NNZ_block_P,
                    " = ",
                    100 * P.count_nonzero() / NNZ_block_P,
                )
                print(
                    "nnz(A) = ",
                    Aosqp.count_nonzero(),
                    " out of ",
                    NNZ_block_A,
                    " = ",
                    100 * Aosqp.count_nonzero() / NNZ_block_A,
                )
            prob = osqp.OSQP()
            prob.setup(
                P,
                q,
                Aosqp,
                losqp,
                uosqp,
                warm_start=False,
                scaling=self.OSQP_scaling,
                max_iter=self.max_qp_iters,
                adaptive_rho=True,
                verbose=self.verboseQP,
                eps_rel=self.eps_rel,
                eps_abs=self.eps_abs,
            )
            t1 = time.time()
            tmp = prob.solve()
            self.qp_time = time.time() - t1
            self.qp_iters = tmp.info.iter
            self.found_qp_sol = tmp.info.status_val == osqp.constant("OSQP_SOLVED")
            # Get solution
            FOUND_NONE = False
            if None in tmp.y:
                FOUND_NONE = True
                self.y_k = np.zeros(tmp.y.shape)
            else:
                self.y_k = tmp.y
            if None in tmp.x:
                FOUND_NONE = True
                res = np.zeros(tmp.x.shape)
            else:
                res = tmp.x
            # If we are benchmarking the QP, check convergence using dual and primal residuals
            # Check here : https://scaron.info/blog/optimality-conditions-and-numerical-tolerances-in-qp-solvers.html
            if self.BENCHMARK:
                m = C.shape[0]
                meq = A.shape[0]
                if FOUND_NONE:
                    self.norm_primal = np.inf
                    self.norm_dual = np.inf
                else:
                    self.norm_primal = np.max(np.abs(A @ res - B))
                    self.norm_dual = np.max(
                        np.abs(
                            P @ res
                            + q
                            + A.T @ self.y_k[:meq]
                            + C.T @ self.y_k[meq : meq + m]
                        )
                    )
                if C.shape[0] > 0:
                    if FOUND_NONE:
                        res_lb = np.inf
                        res_ub = np.inf
                    else:
                        res_lb = np.max(np.abs(np.clip(l - C @ res, 0, np.inf)))
                        res_ub = np.max(np.abs(np.clip(C @ res - u, 0, np.inf)))
                    self.norm_primal = np.maximum(res_lb, self.norm_primal)
                    self.norm_primal = np.maximum(res_ub, self.norm_primal)
                print(f"- Primal residual: {self.norm_primal:.1e}")
                print(f"- Dual residual: {self.norm_dual:.1e}")

        elif self.method == "HPIPM_DENSE":
            # Dimensions
            nv = self.problem.T * (self.ndx + self.nu)  # number of variables
            ne = (
                self.n_eq if self.n_eq is not None else 0
            )  # number of equality constraints
            ng = (
                self.n_in if self.n_in is not None else 0
            )  # number of general (inequality) constraints
            nb = 0  # number of box constraints
            dim = hpipm_python.hpipm_dense_qp_dim()
            dim.set("nv", nv)
            dim.set("nb", nb)
            dim.set("ne", ne)
            dim.set("ng", ng)
            # QP setup
            qp = hpipm_python.hpipm_dense_qp(dim)
            qp.set("H", P)
            qp.set("g", q)
            if ne > 0:
                qp.set("A", A)
                qp.set("b", B)
            if ng > 0:
                qp.set("C", C)
                # need to mask out lb or ub if the box constraints are only one-sided
                # we also mask out infinities (and set the now-irrelevant value to
                # zero), since HPIPM doesn't like infinities
                lg = l
                # import pdb ; pdb.set_trace()
                if lg is not None:
                    # Detect infs and replace them by 0s
                    lg_mask = np.isinf(lg)
                    lg[lg_mask] = 0.0
                    qp.set("lg", lg)
                    # De-activate those constraints
                    qp.set("lg_mask", ~lg_mask)
                else:
                    qp.set("lg_mask", np.zeros(ng, dtype=bool))
                ug = u
                if ug is not None:
                    ug_mask = np.isinf(ug)
                    ug[ug_mask] = 0.0
                    qp.set("ug", ug)
                    qp.set("ug_mask", ~ug_mask)
                else:
                    qp.set("ug_mask", np.zeros(ng, dtype=bool))
            qp_sol = hpipm_python.hpipm_dense_qp_sol(dim)
            # set up solver arg
            # see default args here https://github.com/giaf/hpipm/blob/f7c4f502172b8ea5279c8f4d34afe882407526c3/dense_qp/x_dense_qp_ipm.c#L104
            mode = "speed"
            # create and set default arg based on mode
            arg = hpipm_python.hpipm_dense_qp_solver_arg(dim, mode)
            # create and set default arg based on mode
            arg.set("iter_max", self.max_qp_iters)
            arg.set("tol_comp", self.eps_abs)
            arg.set("tol_eq", self.eps_abs)
            arg.set("tol_ineq", self.eps_abs)
            arg.set("tol_stat", self.eps_abs)
            solver = hpipm_python.hpipm_dense_qp_solver(dim, arg)
            t1 = time.time()
            solver.solve(qp, qp_sol)
            self.qp_time = time.time() - t1
            self.qp_iters = solver.get("iter")
            self.found_qp_sol = solver.get("status") == 0
            # Get solution
            res = qp_sol.get("v").flatten()
            # TODO: retrieve dual solution for SQP
            self.y_k = np.zeros(self.n_in + self.n_eq)
            self.y_k[: self.n_eq] = (
                -qp_sol.get("pi").flatten() if ne > 0 else np.empty((0,))
            )
            # Lagrange multiplier associated with the box constraint can be determined
            # based on complementarity slackness conditions
            lam_lg = qp_sol.get("lam_lg").flatten()
            lam_ug = qp_sol.get("lam_ug").flatten()
            self.y_k[self.n_eq : self.n_eq + self.n_in] = (
                lam_ug if ng > 0 else np.empty((0,))
            )  # This is probably wrong !!!!!
            # If we are benchmarking the QP, check convergence using dual and primal residuals
            # Check here : https://scaron.info/blog/optimality-conditions-and-numerical-tolerances-in-qp-solvers.html
            if self.BENCHMARK:
                # HPIPM residuals
                self.norm_primal = np.maximum(
                    solver.get("max_res_eq"), solver.get("max_res_ineq")
                )
                self.norm_dual = solver.get("max_res_stat")
                print(f"- Primal residual: {self.norm_primal:.1e}")
                print(f"- Dual residual: {self.norm_dual:.1e}")
                # Optionally check that the residual norm returned by HPIPM matches
                # - hand-calculated residual of full QP
                if self.DEBUG:
                    meq = A_qp.shape[0]
                    # Lagrange multipliers for inequality constraint Gx <= h ( a.k.a. "Cx <= ub" and "-Cx <= -lb" )
                    z_qp = np.hstack([lam_ug, lam_lg])
                    self.norm_primal2 = max(
                        [
                            0.0,
                            np.max(G_qp.dot(res) - h_qp) if G_qp.shape[0] > 0 else 0.0,
                            np.linalg.norm(A_qp.dot(res) - b_qp, np.inf)
                            if A_qp is not None
                            else 0.0,
                        ]
                    )
                    self.norm_dual2 = np.linalg.norm(
                        P_qp.dot(res)
                        + q_qp
                        + G_qp.T.dot(z_qp)
                        + A_qp.T.dot(self.y_k[:meq]),
                        np.inf,
                    )
                    if solver.get("status") == 0:
                        assert np.abs(self.norm_primal - self.norm_primal2) < 1e-12
                        assert np.abs(self.norm_dual - self.norm_dual2) < 1e-12
                    print(f"- Primal residual (full) : {self.norm_primal2:.1e}")
                    print(f"- Dual residual (full) : {self.norm_dual2:.1e}")

                    # HPIPM native callbacks
                    if self.VERBOSE_DEBUG:
                        status = solver.get("status")
                        res_stat = solver.get("max_res_stat")
                        res_eq = solver.get("max_res_eq")
                        res_ineq = solver.get("max_res_ineq")
                        res_comp = solver.get("max_res_comp")
                        iters = solver.get("iter")
                        stat = solver.get("stat")
                        print("\nsolver statistics:\n")
                        print("ipm return = {0:1d}\n".format(status))
                        print("ipm max res stat = {:e}\n".format(res_stat))
                        print("ipm max res eq   = {:e}\n".format(res_eq))
                        print("ipm max res ineq = {:e}\n".format(res_ineq))
                        print("ipm max res comp = {:e}\n".format(res_comp))
                        print("ipm iter = {0:1d}\n".format(iters))
                        print("stat =")
                        print(
                            "\titer\talpha_aff\tmu_aff\t\tsigma\t\talpha_prim\talpha_dual\tmu\t\tres_stat\tres_eq\t\tres_ineq\tres_comp"
                        )
                        for ii in range(iters + 1):
                            print(
                                "\t{:d}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}".format(
                                    ii,
                                    stat[ii][0],
                                    stat[ii][1],
                                    stat[ii][2],
                                    stat[ii][3],
                                    stat[ii][4],
                                    stat[ii][5],
                                    stat[ii][6],
                                    stat[ii][7],
                                    stat[ii][8],
                                    stat[ii][9],
                                )
                            )
                        print("")

        elif self.method == "HPIPM_OCP":
            # see here https://docs.acados.org/python_interface/index.html?highlight=nbxe#acados-ocp-solver-interface-acadosocp-and-acadosocpsolver
            # Lagrange multipliers only available here https://github.com/giaf/hpipm/pull/159
            # Dimensions
            ne = (
                self.n_eq if self.n_eq is not None else 0
            )  # number of equality constraints
            ng = (
                self.n_in if self.n_in is not None else 0
            )  # number of general (inequality) constraints
            N = self.problem.T
            dim = hpipm_python.hpipm_ocp_qp_dim(N)
            dim.set("nx", self.ndx, 0, N)
            dim.set("nu", self.nu, 0, N - 1)
            for t in range(N):
                dim.set("ng", self.problem.runningModels[t].ng, t)
            dim.set("ng", self.problem.terminalModel.ng, N)
            # QP setup
            qp = hpipm_python.hpipm_ocp_qp(dim)
            # Fill out OCP running nodes
            for t, (model, data) in enumerate(
                zip(self.problem.runningModels, self.problem.runningDatas)
            ):
                # Dynamics
                if t >= 1:
                    qp.set("A", data.Fx, t)
                qp.set("B", data.Fu, t)
                qp.set("b", self.gap[t], t)
                # Cost
                qp.set("R", data.Luu, t)
                qp.set("r", data.Lu.T, t)
                if t >= 1:
                    qp.set("S", data.Lxu.T, t)
                    qp.set("Q", data.Lxx, t)
                    qp.set("q", data.Lx.T, t)
                # Constraints
                if t >= 1:
                    qp.set("C", data.Gx, t)
                qp.set("D", data.Gu, t)
                # need to mask out lb or ub if the box constraints are only one-sided
                # we also mask out infinities (and set the now-irrelevant value to
                # zero), since HPIPM doesn't like infinities
                lg = model.g_lb.copy()
                if lg is not None:
                    # Detect infs and replace them by 0s
                    lg_mask = np.isinf(lg)
                    lg[lg_mask] = 0.0
                    qp.set("lg", lg - data.g, t)
                    # De-activate those constraints
                    qp.set("lg_mask", ~lg_mask, t)
                else:
                    qp.set("lg_mask", np.zeros(ng, dtype=bool))
                ug = model.g_ub.copy()
                if ug is not None:
                    ug_mask = np.isinf(ug)
                    ug[ug_mask] = 0.0
                    qp.set("ug", ug - data.g, t)
                    qp.set("ug_mask", ~ug_mask, t)
                else:
                    qp.set("ug_mask", np.zeros(ng, dtype=bool))

            # Terminal node
            if self.problem.terminalModel.ng > 0:
                lg = self.problem.terminalModel.g_lb.copy()
                if lg is not None:
                    # Detect infs and replace them by 0s
                    lg_mask = np.isinf(lg)
                    lg[lg_mask] = 0.0
                    qp.set("lg", lg - self.problem.terminalData.g, N)
                    # De-activate those constraints
                    qp.set("lg_mask", ~lg_mask, N)
                else:
                    qp.set("lg_mask", np.zeros(ng, dtype=bool))
                ug = self.problem.terminalModel.g_ub.copy()
                if ug is not None:
                    ug_mask = np.isinf(ug)
                    ug[ug_mask] = 0.0
                    qp.set("ug", ug - self.problem.terminalData.g, N)
                    qp.set("ug_mask", ~ug_mask, N)
                else:
                    qp.set("ug_mask", np.zeros(ng, dtype=bool))
                qp.set("C", self.problem.terminalData.Gx, N)
            qp.set("Q", self.problem.terminalData.Lxx, N)
            qp.set("q", self.problem.terminalData.Lx.T, N)
            qp_sol = hpipm_python.hpipm_ocp_qp_sol(dim)
            # set up solver arg
            mode = "speed"
            # create and set default arg based on mode
            arg = hpipm_python.hpipm_ocp_qp_solver_arg(dim, mode)
            # create and set default arg based on mode
            arg.set("iter_max", self.max_qp_iters)
            arg.set("tol_comp", self.eps_abs)
            arg.set("tol_eq", self.eps_abs)
            arg.set("tol_ineq", self.eps_abs)
            arg.set("tol_stat", self.eps_abs)
            solver = hpipm_python.hpipm_ocp_qp_solver(dim, arg)
            t1 = time.time()
            solver.solve(qp, qp_sol)
            self.qp_time = time.time() - t1
            self.qp_iters = solver.get("iter")
            self.found_qp_sol = solver.get("status") == 0
            # Get solution (need to stack the full QP solution for SQP applications)
            res = np.zeros(self.problem.T * (self.ndx + self.nu))
            self.y_k = np.zeros(self.n_in + self.n_eq)
            self.dx[0] = np.zeros(self.ndx)
            self.lag_mul[0] = np.zeros(self.problem.runningModels[0].state.ndx)
            for t in range(self.problem.T):
                # Stacked (full QP) solution
                res[t * self.ndx : (t + 1) * self.ndx] = qp_sol.get(
                    "x", t + 1
                ).flatten()
                index_u = self.problem.T * self.ndx + t * self.nu
                res[index_u : index_u + self.nu] = qp_sol.get("u", t).flatten()
                # Sequential solution (for KKT computation)
                self.du[t] = qp_sol.get("u", t).flatten()
                self.dx[t + 1] = qp_sol.get("x", t + 1).flatten()
                # Lagrange multipliers associated with dynamics constraint
                if self.DEBUG:
                    self.y_k[t * self.ndx : (t + 1) * self.ndx] = -qp_sol.get(
                        "pi", t
                    ).flatten()
                    self.lag_mul[t + 1] = qp_sol.get("pi", t).flatten()
                    # TODO: retrieve dual solution for SQP
                    # Lagrange multiplier associated with the box constraint can be determined
                    # based on complementarity slackness conditions
                    lam_ug = np.zeros(self.n_in)
                    lam_lg = np.zeros(self.n_in)
                    # lam    = np.zeros(self.n_in)
                    nin_count = 0
                    for t in range(self.problem.T + 1):
                        if t < self.problem.T:
                            model = self.problem.runningModels[t]
                            # data = self.problem.runningDatas[t]
                            # tmp_vec = data.Gu @ self.du[t]
                            # if(t > 0):
                            #     tmp_vec += data.Gx @ self.dx[t]
                        else:
                            model = self.problem.terminalModel
                            # data = self.problem.terminalData
                            # tmp_vec = data.Gx @ self.dx[t]
                        if model.ng > 0:
                            lam_ug[nin_count : nin_count + model.ng] = qp_sol.get(
                                "lam_ug", t
                            ).flatten()
                            lam_lg[nin_count : nin_count + model.ng] = qp_sol.get(
                                "lam_lg", t
                            ).flatten()
                            # # "unified" lagrange multipliers : check constraint residual of the solutiontmp_vec
                            # if(np.linalg.norm(tmp_vec - model.g_lb, np.inf) < 1e-6 and np.linalg.norm(qp_sol.get("lam_ug", t)) < 1e-6):
                            #     lam[nin_count: nin_count+model.ng] = qp_sol.get("lam_lg", t).flatten()
                            # elif(np.linalg.norm(tmp_vec - model.g_lb, np.inf) < 1e-6 and np.linalg.norm(qp_sol.get("lam_lg", t)) < 1e-6):
                            #     lam[nin_count: nin_count+model.ng] = qp_sol.get("lam_ug", t).flatten()
                        nin_count += model.ng
                    self.y_k[self.n_eq : self.n_eq + self.n_in] = (
                        lam_ug  # This is probably wrong !!!!! #TODO use 'lam' instead (needs debug)
                    )
            if self.BENCHMARK:
                # HPIPM residuals
                self.norm_primal = np.maximum(
                    solver.get("max_res_eq"), solver.get("max_res_ineq")
                )
                self.norm_dual = solver.get("max_res_stat")
                print(f"- Primal residual: {self.norm_primal:.1e}")
                print(f"- Dual residual: {self.norm_dual:.1e}")
                # Optionally check that the residual norm returned by HPIPM matches
                # 1) hand-calculated residual of full QP
                # 2) KKT criteria (sequential form)
                if self.DEBUG:
                    # 1) Full QP
                    # Lagrange multipliers for inequality constraint Gx <= h ( a.k.a. "Cx <= ub" and "-Cx <= -lb" )
                    meq = A_qp.shape[0]
                    self.norm_primal2 = max(
                        [
                            0.0,
                            max(G_qp @ res - h_qp) if G_qp.shape[0] > 0 else 0.0,
                            max(A_qp @ res - b_qp) if A_qp.shape[0] > 0 else 0.0,
                        ]
                    )
                    self.norm_dual2 = np.linalg.norm(
                        P_qp.dot(res)
                        + q_qp
                        + C.T @ lam_ug
                        - C.T @ lam_lg
                        + A_qp.T.dot(self.y_k[:meq]),
                        np.inf,
                    )
                    if solver.get("status") == 0:
                        assert (
                            np.linalg.norm(self.norm_primal - self.norm_primal2) < 1e-12
                        )
                        assert np.linalg.norm(self.norm_dual - self.norm_dual2) < 1e-12
                    print(
                        f"- [DEBUG] Primal residual (full QP) : {self.norm_primal2:.1e}"
                    )
                    print(f"- [DEBUG] Dual residual (full QP) : {self.norm_dual2:.1e}")

                    # 2) Check sequential KKT residual
                    KKT = 0
                    l1 = 0
                    l2 = 0
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
                                + Cu.T @ qp_sol.get("lam_ug", t).flatten()
                                - Cu.T @ qp_sol.get("lam_lg", t).flatten()
                            )
                            KKT = max(KKT, max(abs(lu)))
                            continue
                        lx = (
                            data.Lxx @ self.dx[t]
                            + data.Lxu @ self.du[t]
                            + data.Lx
                            + data.Fx.T @ self.lag_mul[t + 1]
                            - self.lag_mul[t]
                            + Cx.T @ qp_sol.get("lam_ug", t).flatten()
                            - Cx.T @ qp_sol.get("lam_lg", t).flatten()
                        )
                        lu = (
                            data.Luu @ self.du[t]
                            + data.Lxu.T @ self.dx[t]
                            + data.Lu
                            + data.Fu.T @ self.lag_mul[t + 1]
                            + Cu.T @ qp_sol.get("lam_ug", t).flatten()
                            - Cu.T @ qp_sol.get("lam_lg", t).flatten()
                        )
                        KKT = max(KKT, max(abs(lx)), max(abs(lu)))
                        if model.ng != 0:
                            l1 = np.max(
                                np.abs(
                                    np.clip(
                                        model.g_lb
                                        - Cx @ self.dx[t]
                                        - Cu @ self.du[t]
                                        - data.g,
                                        0,
                                        np.inf,
                                    )
                                )
                            )
                            l2 = np.max(
                                np.abs(
                                    np.clip(
                                        Cx @ self.dx[t]
                                        + Cu @ self.du[t]
                                        + data.g
                                        - model.g_ub,
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
                            np.abs(
                                np.clip(
                                    model.g_lb - Cx @ self.dx[-1] - data.g, 0, np.inf
                                )
                            )
                        )
                        l2 = np.max(
                            np.abs(
                                np.clip(
                                    Cx @ self.dx[-1] + data.g - model.g_ub, 0, np.inf
                                )
                            )
                        )
                    lx = (
                        self.problem.terminalData.Lxx @ self.dx[-1]
                        + self.problem.terminalData.Lx
                        - self.lag_mul[-1]
                        + Cx.T @ qp_sol.get("lam_ug", N).flatten()
                        - Cx.T @ qp_sol.get("lam_lg", N).flatten()
                    )
                    KKT = max(KKT, l1, l2)
                    KKT = max(KKT, max(abs(lx)))
                    assert (
                        np.linalg.norm(KKT - max(self.norm_primal, self.norm_dual))
                        < 1e-12
                    )
                    print(f"- [DEBUG] KKT residual (sequential form) : {KKT:.1e}")

                    # HPIPM native callbacks
                    if self.DEBUG_VERBOSE:
                        # get solver statistics
                        # 'lam' returned by HPIPM print is equal to ( lam_lg[1], lam_ug[1], ..., lam_lg[N], lam_ug[N] )
                        # qp_sol.print_C_struct()
                        status = solver.get("status")
                        res_stat = solver.get("max_res_stat")
                        res_eq = solver.get("max_res_eq")
                        res_ineq = solver.get("max_res_ineq")
                        res_comp = solver.get("max_res_comp")
                        iters = solver.get("iter")
                        stat = solver.get("stat")
                        print("\nsolver statistics:\n")
                        print("ipm return = {0:1d}\n".format(status))
                        print("ipm max res stat = {:e}\n".format(res_stat))
                        print("ipm max res eq   = {:e}\n".format(res_eq))
                        print("ipm max res ineq = {:e}\n".format(res_ineq))
                        print("ipm max res comp = {:e}\n".format(res_comp))
                        print("ipm iter = {0:1d}\n".format(iters))
                        print("stat =")
                        print(
                            "\titer\talpha_aff\tmu_aff\t\tsigma\t\talpha_prim\talpha_dual\tmu\t\tres_stat\tres_eq\t\tres_ineq\tres_comp"
                        )
                        for ii in range(iters + 1):
                            print(
                                "\t{:d}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}".format(
                                    ii,
                                    stat[ii][0],
                                    stat[ii][1],
                                    stat[ii][2],
                                    stat[ii][3],
                                    stat[ii][4],
                                    stat[ii][5],
                                    stat[ii][6],
                                    stat[ii][7],
                                    stat[ii][8],
                                    stat[ii][9],
                                )
                            )
                        print("")

        elif self.method == "CustomOSQP":
            Aeq = sparse.csr_matrix(A)
            Aineq = sparse.csr_matrix(C)
            self.Aosqp = sparse.vstack([Aeq, Aineq])

            self.losqp = np.hstack([B, l])
            self.uosqp = np.hstack([B, u])

            self.P = P
            self.q = np.array(q)

            self.x_k = np.zeros(self.n_vars)
            self.z_k = np.zeros(self.n_in + self.n_eq)
            self.y_k = np.zeros(self.n_in + self.n_eq)
            t0 = time.time()
            res = self.optimize_osqp(maxiters=self.max_qp_iters)
            self.qp_time = time.time() - t0
            self.found_qp_sol = self.converged
            self.norm_primal = self.r_prim
            self.norm_dual = self.r_dual

        elif self.method == "StagewiseQPKKT":
            self.A_eq = sparse.csr_matrix(A.copy())
            self.A_in = sparse.csr_matrix(C.copy())
            self.b = B.copy()
            self.lboyd = l.copy()
            self.uboyd = u.copy()

            self.P = P.copy()
            self.q = np.array(q).copy()

            if self.initialize:
                self.xs_vec = np.array(self.xs).flatten()[self.ndx :]
                self.us_vec = np.array(self.us).flatten()
                self.x_k = np.zeros_like(np.hstack((self.xs_vec, self.us_vec)))
                self.z_k = np.zeros(self.n_in)
                self.y_k = np.zeros(self.n_in)

                self.initialize = False

            res = self.optimize_boyd(maxiters=self.max_qp_iters)
            for t in range(self.problem.T):
                self.lag_mul[t + 1] = -self.v_k_1[t * self.ndx : (t + 1) * self.ndx]

            nin_count = 0
            for t in range(self.problem.T + 1):
                if t < self.problem.T:
                    model = self.problem.runningModels[t]
                else:
                    model = self.problem.terminalModel
                if model.ng == 0:
                    continue
                self.y[t] = self.y_k[nin_count : nin_count + model.ng]
                nin_count += model.ng

        if (
            self.method == "CustomOSQP"
            or self.method == "OSQP"
            or self.method == "HPIPM_DENSE"
            or self.method == "HPIPM_OCP"
        ):
            nin_count = self.n_eq
            self.lag_mul[0] = np.zeros(self.problem.runningModels[0].state.ndx)
            for t in range(self.problem.T + 1):
                if t < self.problem.T:
                    model = self.problem.runningModels[t]
                else:
                    model = self.problem.terminalModel
                if t < self.problem.T:
                    self.lag_mul[t + 1] = -self.y_k[t * self.ndx : (t + 1) * self.ndx]
                if model.ng == 0:
                    continue
                self.y[t] = self.y_k[nin_count : nin_count + model.ng]
                nin_count += model.ng

        self.dx[0] = np.zeros(self.ndx)
        for t in range(self.problem.T):
            self.dx[t + 1] = res[t * self.ndx : (t + 1) * self.ndx]
            index_u = self.problem.T * self.ndx + t * self.nu
            self.du[t] = res[index_u : index_u + self.nu]

        self.x_grad_norm = np.linalg.norm(self.dx) / (self.problem.T + 1)
        self.u_grad_norm = np.linalg.norm(self.du) / self.problem.T

    def acceptStep(self, alpha):
        for t, (model, data) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha * self.dx[t])
            self.us_try[t] = self.us[t] + alpha * self.du[t]
        self.xs_try[-1] = model.state.integrate(
            self.xs[-1], alpha * self.dx[-1]
        )  ## terminal state update

        self.setCandidate(self.xs_try, self.us_try, False)

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
        self.computeDirectionFullQP(KKT=False)
        self.acceptStep(alpha=1.0)
        # self.reset_params()

    def allocateDataQP(self):
        #
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]
        #
        self.dx = [np.zeros(m.state.ndx) for m in self.models()]
        self.du = [np.zeros(m.nu) for m in self.problem.runningModels]
        #
        self.y = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]
        self.lag_mul = [np.zeros(m.state.ndx) for m in self.models()]
        #
        self.dz_relaxed = [np.zeros(m.ng) for m in self.problem.runningModels] + [
            np.zeros(self.problem.terminalModel.ng)
        ]
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
        self.nx = self.problem.terminalModel.state.nx
        self.ndx = self.problem.terminalModel.state.ndx
        self.nu = self.problem.runningModels[0].nu

        n_eq = (
            sum([m.nh for m in self.problem.runningModels])
            + self.problem.terminalModel.nh
        )

        assert n_eq == 0
