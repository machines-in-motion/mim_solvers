## This contains various QP solvers that solve the convex subproblem of the sqp
## Author : Avadesh Meduri & Armand Jordana
## Date : 21/03/2022

import numpy as np
import osqp
import proxsuite
import time 
from scipy import sparse
from . py_osqp import CustomOSQP
from . stagewise_qp_kkt import StagewiseQPKKT
from crocoddyl import SolverAbstract
import hpipm_python
import time 
# from qpsolvers import solve_problem, Problem, Solution

class QPSolvers(SolverAbstract, CustomOSQP, StagewiseQPKKT):

    def __init__(self, shootingProblem, method, verboseQP = True):
        
        SolverAbstract.__init__(self, shootingProblem)        

        assert method == "ProxQP" or method=="OSQP"\
              or method=="CustomOSQP" or method =="StagewiseQPKKT"\
              or method=="HPIPM_dense" or method=='HPIPM_ocp'
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
        self.eps_rel = 0.

        self.BENCHMARK = True

        if self.verboseQP:
            print("USING " + str(method))

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  
        
    def calc(self, recalc = True):
        # compute cost and derivatives at deterministic nonlinear trajectory 
        if recalc:
            self.problem.calc(self.xs, self.us)
            self.problem.calcDiff(self.xs, self.us)
        self.cost = 0
        self.constraint_norm = 0

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            # Inequality
            self.constraint_norm += np.linalg.norm(np.clip(model.g_lb - data.g, 0, np.inf), 1) 
            self.constraint_norm += np.linalg.norm(np.clip(data.g - model.g_ub, 0, np.inf), 1)
            # Equality
            # self.constraint_norm += np.linalg.norm(data.h - model.h_ub, 0, np.inf), 1)


        self.constraint_norm += np.linalg.norm(np.clip(self.problem.terminalModel.g_lb - self.problem.terminalData.g, 0, np.inf), 1) 
        self.constraint_norm += np.linalg.norm(np.clip(self.problem.terminalData.g - self.problem.terminalModel.g_ub, 0, np.inf), 1)


        for t, (model, data) in enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            self.gap[t] = model.state.diff(self.xs[t+1].copy(), data.xnext.copy()) #gaps
            self.cost += data.cost


        self.gap_norm = sum(np.linalg.norm(self.gap.copy(), 1, axis = 1))
        self.cost += self.problem.terminalData.cost 
        self.gap = self.gap.copy()


    def computeDirectionFullQP(self):
        self.n_vars  = self.problem.T*(self.ndx + self.nu)

        P = np.zeros((self.problem.T*(self.ndx + self.nu), self.problem.T*(self.ndx + self.nu)))
        q = np.zeros(self.problem.T*(self.ndx + self.nu))
        
        Asize = self.problem.T*(self.ndx + self.nu)
        A = np.zeros((self.problem.T*self.ndx, Asize))
        B = np.zeros(self.problem.T*self.ndx)

        NNZ_block_P = 0
        NNZ_block_A = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            index_u = self.problem.T*self.ndx + t * self.nu
            if t>=1:
                index_x = (t-1) * self.ndx
                P[index_x:index_x+self.ndx, index_x:index_x+self.ndx] = data.Lxx.copy()
                NNZ_block_P += np.size(data.Lxx)
                P[index_x:index_x+self.ndx, index_u:index_u+self.nu] = data.Lxu.copy()
                NNZ_block_P += np.size(data.Lxu)
                P[index_u:index_u+self.nu, index_x:index_x+self.ndx] = data.Lxu.T.copy()
                NNZ_block_P += np.size(data.Lxu.T)
                q[index_x:index_x+self.ndx] = data.Lx.copy()

            P[index_u:index_u+self.nu, index_u:index_u+self.nu] = data.Luu.copy()
            NNZ_block_P += np.size(data.Luu)
            q[index_u:index_u+self.nu] = data.Lu.copy()
            
            A[t * self.ndx: (t+1) * self.ndx, index_u:index_u+self.nu] = - data.Fu.copy() 
            NNZ_block_A += np.size(data.Fu)
            A[t * self.ndx: (t+1) * self.ndx, t * self.ndx: (t+1) * self.ndx] = np.eye(self.ndx)
            NNZ_block_A += self.ndx

            if t >=1:
                A[t * self.ndx: (t+1) * self.ndx, (t-1) * self.ndx: t * self.ndx] = - data.Fx.copy()
                NNZ_block_A += np.size(data.Fx) 
            B[t * self.ndx: (t+1) * self.ndx] = self.gap[t].copy()


        P[(self.problem.T-1)*self.ndx:self.problem.T*self.ndx, self.problem.T*self.ndx-self.ndx:self.problem.T*self.ndx] = self.problem.terminalData.Lxx.copy()
        NNZ_block_P += np.size(self.problem.terminalData.Lxx)
        q[(self.problem.T-1)*self.ndx:self.problem.T*self.ndx] = self.problem.terminalData.Lx.copy()


        n = self.problem.T*(self.ndx + self.nu)
        self.n_eq = self.problem.T*self.ndx

        self.n_in = sum([len(self.y[i]) for i in range(len(self.y))])

        C = np.zeros((self.n_in, self.problem.T*(self.ndx + self.nu)))
        l = np.zeros(self.n_in)
        u = np.zeros(self.n_in)

        nin_count = 0
        index_x = self.problem.T*self.ndx
        
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            if model.ng == 0:
                continue
            l[nin_count: nin_count + model.ng] = model.g_lb - data.g
            u[nin_count: nin_count + model.ng] = model.g_ub - data.g
            if t > 0:
                C[nin_count: nin_count + model.ng, (t-1)*self.ndx: t*self.ndx] = data.Gx
                NNZ_block_A += np.size(data.Gx)
            C[nin_count: nin_count + model.ng, index_x+t*self.nu: index_x+(t+1)*self.nu] = data.Gu
            NNZ_block_A += np.size(data.Gu)
            nin_count += model.ng

        model = self.problem.terminalModel
        data = self.problem.terminalData
        if model.ng != 0:
            l[nin_count: nin_count + model.ng] = model.g_lb - data.g
            u[nin_count: nin_count + model.ng] = model.g_ub - data.g
            C[nin_count: nin_count + model.ng, (self.problem.T-1)*self.ndx: self.problem.T*self.ndx] = data.Gx
            NNZ_block_A += np.size(data.Gx)
            nin_count += model.ng


        # # If we are benchmarking, setup the problem using QPSolvers (S. Caron)
        # # in order to get the unified convergence metrics (prim_res, dual_res, dual_gap)
        # if(self.BENCHMARK):
        #     # We follow the convention of QPSolvers package, i.e.
        #     # min_x (1/2) x^T P x + q^T x
        #     #   s.t. Ax  = b
        #     #        Gx <= h
        #     # see here https://github.com/qpsolvers/qpsolvers/blob/main/qpsolvers/problem.py
        #     P_qp = P ; q_qp = q
        #     A_qp = A ; b_qp = B
        #     G_qp = np.vstack([C, -C])
        #     h_qp = np.hstack([u, -l]) 
        #     prob_qp = Problem(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp) # no box constraints
        #     sol_qp = Solution(prob_qp)

        if self.method == "ProxQP":
            qp = proxsuite.proxqp.sparse.QP(n, self.n_eq, self.n_in)
            qp.settings.eps_abs = self.eps_abs
            qp.settings.eps_rel = self.eps_rel
            # qp.settings.max_iter = 100
            # qp.settings.max_iter_in = 100
            qp.init(P, q, A, B, C, l, u)      

            print("start PROXQP")
            t1 = time.time()
            qp.solve()
            print("PROXQP time : ", time.time() - t1)
            res = qp.results.x
            self.z_k = qp.results.z
            self.y_k = qp.results.y
            self.qp_iters = qp.results.info.iter

            # KKT1 = np.max(np.abs(P @ res + q + A.T @ self.y_k  + C.T @ self.z_k))
            # KKT2 = np.max(np.abs(A @ res - B))
            # KKT3 = np.max(np.abs(np.clip(l - C @ res, 0, np.inf)))
            # KKT4 =  np.max(np.abs(np.clip(C @ res - u, 0, np.inf)))
            # print("prox chek", max([KKT1, KKT2, KKT3, KKT4]) )
            
            for t in range(self.problem.T):
                self.lag_mul[t+1] = - qp.results.y[t * self.ndx: (t+1) * self.ndx] 
            nin_count = 0
            for t in range(self.problem.T+1):
                if t < self.problem.T:
                    model = self.problem.runningModels[t]
                else:
                    model = self.problem.terminalModel
                if model.ng == 0:
                    continue
                self.y[t] = qp.results.z[nin_count:nin_count + model.ng]
                nin_count += model.ng          
            
        elif self.method == "OSQP":
            Aeq = sparse.csr_matrix(A)
            Aineq = sparse.csr_matrix(C)
            Aosqp = sparse.vstack([Aeq, Aineq])
            losqp = np.hstack([B, l])
            uosqp = np.hstack([B, u])
            P = sparse.csr_matrix(P)
            # print("nnz(P) = ", P.count_nonzero(), " out of ", NNZ_block_P, " = ", 100* P.count_nonzero() / NNZ_block_P)
            # print("nnz(A) = ", Aosqp.count_nonzero(), " out of ", NNZ_block_A, " = ", 100* Aosqp.count_nonzero() / NNZ_block_A)
            prob = osqp.OSQP()
            prob.setup(P, q, Aosqp, losqp, uosqp, warm_start=False, scaling=False,  max_iter = self.max_qp_iters, \
                            adaptive_rho=True, verbose = self.verboseQP, eps_rel=self.eps_rel, eps_abs=self.eps_abs)     
            print("start OSQP")
            t1 = time.time()
            tmp = prob.solve()
            print("HPIPM time : ", time.time() - t1)
            # self.qp_time = time.time() - t1
            res = tmp.x
            self.y_k = tmp.y
            self.qp_iters = tmp.info.iter
            # If we are benchmarking, compute the metrics using qpsolvers package
            # see here how the Solution class is populated with OSQP : https://github.com/qpsolvers/qpsolvers/blob/main/qpsolvers/solvers/osqp_.py
            # if(self.BENCHMARK):
            #     sol_qp.found = tmp.info.status_val == osqp.constant("OSQP_SOLVED")
            #     if(sol_qp.found):
            #         sol_qp.x = tmp.x
            #         m = C.shape[0] if G_qp is not None else 0 ; meq = A_qp.shape[0] if A_qp is not None else 0
            #         # Lagrange multipliers for equality constraint Ax = b
            #         sol_qp.y = tmp.y[:meq] if A_qp is not None else np.empty((0,))
            #         # Lagrange multipliers for inequality constraint Gx <= h ( a.k.a. "Cx <= ub" and "-Cx <= -lb" )
            #         z = tmp.y[meq:meq+m] if G_qp is not None else np.empty((0,))
            #         zp = np.maximum(z, np.zeros_like(z)) # see OSQP paper Eq. (9)
            #         zm = np.minimum(z, np.zeros_like(z)) # see OSQP paper Eq. (9)
            #         sol_qp.z = np.hstack([zp, zm])
            #         # No box constraints in the present formulation
            #         sol_qp.z_box = np.empty((0,))
            #         # Print convergence metrics
            #         self.is_optimal      = sol_qp.is_optimal(self.eps_abs)
            #         self.primal_residual = sol_qp.primal_residual()
            #         self.dual_residual   = sol_qp.dual_residual()
            #         self.duality_gap     = sol_qp.duality_gap()
            #         print(f"- Solution is{'' if sol_qp.is_optimal(1e-3) else ' NOT'} optimal")
            #         print(f"- Primal residual: {sol_qp.primal_residual():.1e}")
            #         print(f"- Dual residual: {sol_qp.dual_residual():.1e}")
            #         print(f"- Duality gap: {sol_qp.duality_gap():.1e}")
            #     else:
            #         self.is_optimal      = False
            #         self.primal_residual = np.inf
            #         self.dual_residual   = np.inf
            #         self.duality_gap     = np.inf

        elif self.method == "HPIPM_dense":
            # Dimensions
            nv = self.problem.T*(self.ndx + self.nu)         # number of variables
            ne = self.n_eq if self.n_eq is not None else 0   # number of equality constraints
            ng = self.n_in if self.n_in is not None else 0   # number of general (inequality) constraints
            nb = 0                                           # number of box constraints
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
            mode = 'speed'
            # create and set default arg based on mode
            arg = hpipm_python.hpipm_dense_qp_solver_arg(dim, mode)
            # create and set default arg based on mode
            arg.set('iter_max', self.max_qp_iters)
            arg.set('tol_comp', 1)
            arg.set('tol_eq', self.eps_abs)
            arg.set('tol_ineq', self.eps_abs)
            arg.set('tol_stat', self. eps_abs)
            solver = hpipm_python.hpipm_dense_qp_solver(dim, arg)
            print("start HPIPM")
            t1 = time.time()
            solver.solve(qp, qp_sol)
            print("HPIPM time : ", time.time() - t1)
            VERBOSE = False
            if(VERBOSE):
                v = qp_sol.get('v')
                pi = qp_sol.get('pi')
                lam_lb = qp_sol.get('lam_lb')
                lam_ub = qp_sol.get('lam_ub')
                lam_lg = qp_sol.get('lam_lg')
                lam_ug = qp_sol.get('lam_ug')
                print('v      = {}'.format(v.flatten()))
                print('pi     = {}'.format(pi.flatten()))
                print('lam_lb = {}'.format(lam_lb.flatten()))
                print('lam_ub = {}'.format(lam_ub.flatten()))
                print('lam_lg = {}'.format(lam_lg.flatten()))
                print('lam_ug = {}'.format(lam_ug.flatten()))
                # get solver statistics
                status = solver.get('status')
                res_stat = solver.get('max_res_stat')
                res_eq = solver.get('max_res_eq')
                res_ineq = solver.get('max_res_ineq')
                res_comp = solver.get('max_res_comp')
                iters = solver.get('iter')
                stat = solver.get('stat')
                print('\nsolver statistics:\n')
                print('ipm return = {0:1d}\n'.format(status))
                print('ipm max res stat = {:e}\n'.format(res_stat))
                print('ipm max res eq   = {:e}\n'.format(res_eq))
                print('ipm max res ineq = {:e}\n'.format(res_ineq))
                print('ipm max res comp = {:e}\n'.format(res_comp))
                print('ipm iter = {0:1d}\n'.format(iters))
                print('stat =')
                print('\titer\talpha_aff\tmu_aff\t\tsigma\t\talpha_prim\talpha_dual\tmu\t\tres_stat\tres_eq\t\tres_ineq\tres_comp')
                for ii in range(iters+1):
                    print('\t{:d}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}'.format(ii, stat[ii][0], stat[ii][1], stat[ii][2], stat[ii][3], stat[ii][4], stat[ii][5], stat[ii][6], stat[ii][7], stat[ii][8], stat[ii][9]))
                print('')
            res = qp_sol.get('v').flatten()
            self.y_k = np.zeros(self.n_in + self.n_eq)
            self.y_k[:self.n_eq] = -qp_sol.get("pi").flatten() if ne > 0 else np.empty((0,))
            self.y_k[self.n_eq:self.n_eq + self.n_in] = qp_sol.get("lam_ug").flatten() if ng > 0 else np.empty((0,))
            self.qp_iters = solver.get("iter")

        elif self.method == "HPIPM_ocp":
            # Dimensions
            N = self.problem.T
            dim = hpipm_python.hpipm_ocp_qp_dim(N)
            dim.set("nx", self.nx, 0, N)
            dim.set("nu", self.nu, 0, N-1)
            # QP setup
            qp = hpipm_python.hpipm_ocp_qp(dim)
            # Fill out OCP running nodes
            print("horizon = ", N)
            for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # Dynamics
                print("node ", t, "->", t+1)
                qp.set('A', data.Fx, t, t+1)
                qp.set('B', data.Fu, t, t+1)
                qp.set('b', self.gap[t], t, t+1)
                # Cost 
                qp.set('S', data.Lxu.T, t, t+1)
                qp.set('R', data.Luu, t, t+1)
                qp.set('r', data.Lu, t, t+1)
                qp.set('Q', data.Lxx, t, t+1)
                qp.set('q', data.Lx, t, t+1)
                # Constraints
                if(t>=1):
                    qp.set('C', data.Gx, t, t+1)
                qp.set('D', data.Gu, t, t+1)
                qp.set('lg', model.g_lb - data.g, 0)
                qp.set('ug', model.g_lb - data.g, 0)
            # Terminal node
            qp.set('Q', self.problem.terminalData.Lxx, N, N+1)
            qp.set('q', self.problem.terminalData.Lx, N, N+1)
            # qp.set('C_e', self.problem.terminalData.Gx, N)
            # qp.set('lg_e', self.problem.terminalModel.g_lb - self.problem.terminalData.g, 0)
            # qp.set('ug_e', self.problem.terminalModel.g_lb - self.problem.terminalData.g, 0)
            # qp.set('Jbu', self.problem.terminalData.Gu, N, N+1)
            
            qp_sol = hpipm_python.hpipm_dense_qp_sol(dim)
            # set up solver arg
            mode = 'speed'
            # create and set default arg based on mode
            arg = hpipm_python.hpipm_dense_qp_solver_arg(dim, mode)
            # create and set default arg based on mode
            arg.set('iter_max', self.max_qp_iters)
            arg.set('tol_comp', 1)
            arg.set('tol_eq', self.eps_abs)
            arg.set('tol_ineq', self.eps_abs)
            arg.set('tol_stat', self. eps_abs)
            solver = hpipm_python.hpipm_dense_qp_solver(dim, arg)
            print("start HPIPM")
            t1 = time.time()
            solver.solve(qp, qp_sol)
            print("HPIPM time : ", time.time() - t1)
            VERBOSE = False
            if(VERBOSE):
                v = qp_sol.get('v')
                pi = qp_sol.get('pi')
                lam_lb = qp_sol.get('lam_lb')
                lam_ub = qp_sol.get('lam_ub')
                lam_lg = qp_sol.get('lam_lg')
                lam_ug = qp_sol.get('lam_ug')
                print('v      = {}'.format(v.flatten()))
                print('pi     = {}'.format(pi.flatten()))
                print('lam_lb = {}'.format(lam_lb.flatten()))
                print('lam_ub = {}'.format(lam_ub.flatten()))
                print('lam_lg = {}'.format(lam_lg.flatten()))
                print('lam_ug = {}'.format(lam_ug.flatten()))
                # get solver statistics
                status = solver.get('status')
                res_stat = solver.get('max_res_stat')
                res_eq = solver.get('max_res_eq')
                res_ineq = solver.get('max_res_ineq')
                res_comp = solver.get('max_res_comp')
                iters = solver.get('iter')
                stat = solver.get('stat')
                print('\nsolver statistics:\n')
                print('ipm return = {0:1d}\n'.format(status))
                print('ipm max res stat = {:e}\n'.format(res_stat))
                print('ipm max res eq   = {:e}\n'.format(res_eq))
                print('ipm max res ineq = {:e}\n'.format(res_ineq))
                print('ipm max res comp = {:e}\n'.format(res_comp))
                print('ipm iter = {0:1d}\n'.format(iters))
                print('stat =')
                print('\titer\talpha_aff\tmu_aff\t\tsigma\t\talpha_prim\talpha_dual\tmu\t\tres_stat\tres_eq\t\tres_ineq\tres_comp')
                for ii in range(iters+1):
                    print('\t{:d}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}\t{:e}'.format(ii, stat[ii][0], stat[ii][1], stat[ii][2], stat[ii][3], stat[ii][4], stat[ii][5], stat[ii][6], stat[ii][7], stat[ii][8], stat[ii][9]))
                print('')
            res = qp_sol.get('v').flatten()
            self.y_k = np.zeros(self.n_in + self.n_eq)
            self.y_k[:self.n_eq] = -qp_sol.get("pi").flatten() if ne > 0 else np.empty((0,))
            self.y_k[self.n_eq:self.n_eq + self.n_in] = qp_sol.get("lam_ug").flatten() if ng > 0 else np.empty((0,))
            self.qp_iters = solver.get("iter")


        elif self.method == "CustomOSQP" :
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
            res = self.optimize_osqp(maxiters=self.max_qp_iters)

        elif self.method == "StagewiseQPKKT":
            self.A_eq = sparse.csr_matrix(A.copy())
            self.A_in = sparse.csr_matrix(C.copy())
            self.b = B.copy()
            self.lboyd = l.copy()
            self.uboyd = u.copy()

            self.P = P.copy()
            self.q = np.array(q).copy()
            
            if self.initialize:
                self.xs_vec = np.array(self.xs).flatten()[self.ndx:]
                self.us_vec = np.array(self.us).flatten()
                self.x_k = np.zeros_like(np.hstack((self.xs_vec, self.us_vec)))
                self.z_k = np.zeros(self.n_in)
                self.y_k = np.zeros(self.n_in)

                self.initialize = False

            res = self.optimize_boyd(maxiters=self.max_qp_iters)
            for t in range(self.problem.T):
                self.lag_mul[t+1] = - self.v_k_1[t * self.ndx: (t+1) * self.ndx] 
            
            nin_count= 0
            for t in range(self.problem.T+1):
                if t < self.problem.T:
                    model = self.problem.runningModels[t]
                else:
                    model = self.problem.terminalModel
                if model.ng == 0:
                    continue
                self.y[t] = self.y_k[nin_count:nin_count + model.ng]
                nin_count += model.ng

        if self.method == "CustomOSQP" or self.method == "OSQP" or self.method == 'HPIPM_dense':
            nin_count = self.n_eq
            self.lag_mul[0] = np.zeros(self.problem.runningModels[0].state.ndx)
            for t in range(self.problem.T+1):
                if t < self.problem.T:
                    model = self.problem.runningModels[t]
                else:
                    model = self.problem.terminalModel
                if t < self.problem.T:
                    self.lag_mul[t+1] = - self.y_k[t * self.ndx: (t+1) * self.ndx]
                if model.ng == 0:
                    continue
                self.y[t] = self.y_k[nin_count:nin_count + model.ng]
                nin_count += model.ng

        self.dx[0] = np.zeros(self.ndx)
        for t in range(self.problem.T):
            self.dx[t+1] = res[t * self.ndx: (t+1) * self.ndx] 
            index_u = self.problem.T*self.ndx + t * self.nu
            self.du[t] = res[index_u:index_u+self.nu]

        self.x_grad_norm = np.linalg.norm(self.dx)/(self.problem.T+1)
        self.u_grad_norm = np.linalg.norm(self.du)/self.problem.T

    def acceptStep(self, alpha):
        
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t] = self.us[t] + alpha*self.du[t]    
        self.xs_try[-1] = model.state.integrate(self.xs[-1], alpha*self.dx[-1]) ## terminal state update

        self.setCandidate(self.xs_try, self.us_try, False)

    def solve(self, init_xs=None, init_us=None, maxiter=1000, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        self.max_qp_iters = maxiter
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        self.setCandidate(init_xs, init_us, False)
        self.computeDirectionFullQP(KKT=False)
        self.acceptStep(alpha = 1.0)
        # self.reset_params()
        
    def allocateDataQP(self):    
        #
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        #
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        #
        self.y = [np.zeros(m.ng) for m in self.problem.runningModels] + [np.zeros(self.problem.terminalModel.ng)] 
        self.lag_mul = [np.zeros(m.state.ndx) for m  in self.models()] 
        #
        self.dz_relaxed = [np.zeros(m.ng) for m in self.problem.runningModels] + [np.zeros(self.problem.terminalModel.ng)] 
        #
        self.x_grad = [np.zeros(m.state.ndx) for m in self.models()]
        self.u_grad = [np.zeros(m.nu) for m in self.problem.runningModels]

        self.gap = [np.zeros(m.state.ndx) for m in self.models()] # gaps
        self.gap_try = [np.zeros(m.state.ndx) for m in self.models()] # gaps for line search

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


        n_eq = sum([m.nh for m in self.problem.runningModels]) + self.problem.terminalModel.nh

        assert n_eq == 0