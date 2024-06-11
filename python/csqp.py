## Implementation of the sequential constrained lqr
## Author : Avadesh Meduri and Armand Jordana
## Date : 9/3/2022

import numpy as np
import scipy.linalg as scl
from qp_solvers.stagewise_qp import StagewiseADMM
from qp_solvers.qpsolvers import QPSolvers
import collections
pp = lambda s : np.format_float_scientific(s, exp_digits=2, precision =4)
from qpsolvers import Problem, Solution

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class CSQP(StagewiseADMM, QPSolvers):

    def __init__(self, shootingProblem, method, use_filter_line_search=True, with_callbacks = False, qp_with_callbacks = False):

        if method == "StagewiseQP":
            StagewiseADMM.__init__(self, shootingProblem, verboseQP = qp_with_callbacks)
            self.using_qp = 0        
        else:
            QPSolvers.__init__(self, shootingProblem, method, verboseQP = qp_with_callbacks)
            self.using_qp = 1        

        self.mu1 = 1e1
        self.mu2 = 1e1
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
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t] = self.us[t] + alpha*self.du[t]    
        self.xs_try[-1] = model.state.integrate(self.xs[-1], alpha*self.dx[-1]) ## terminal state update


        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):

            model.calc(data, self.xs_try[t], self.us_try[t])  

            self.gap_try[t] = model.state.diff(self.xs_try[t+1], data.xnext) #gaps
            self.cost_try += data.cost

            self.constraint_norm_try += np.linalg.norm(np.clip(model.g_lb - data.g, 0, np.inf), 1) 
            self.constraint_norm_try += np.linalg.norm(np.clip(data.g - model.g_ub, 0, np.inf), 1)


        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])  
        self.constraint_norm_try += np.linalg.norm(np.clip(self.problem.terminalModel.g_lb - self.problem.terminalData.g, 0, np.inf), 1) 
        self.constraint_norm_try += np.linalg.norm(np.clip(self.problem.terminalData.g - self.problem.terminalModel.g_ub, 0, np.inf), 1)

        self.cost_try += self.problem.terminalData.cost


        self.gap_norm_try = sum(np.linalg.norm(self.gap_try, 1, axis = 1))
        self.merit_try =  self.cost_try + self.mu1 * self.gap_norm_try + self.mu2 * self.constraint_norm_try
   
    def LQ_problem_KKT_check(self):
        KKT = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            Cx, Cu = data.Gx, data.Gu
            if t == 0:
                lu = data.Luu @ self.du[t] + data.Lxu.T @ self.dx[t] + data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
                KKT = max(KKT, max(abs(lu)))
                continue

            lx = data.Lxx @ self.dx[t] + data.Lxu @ self.du[t] + data.Lx + data.Fx.T @ self.lag_mul[t+1] - self.lag_mul[t] + Cx.T @ self.y[t]
            lu = data.Luu @ self.du[t] + data.Lxu.T @ self.dx[t] + data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
            KKT = max(KKT, max(abs(lx)), max(abs(lu)))

            l1 = np.max(np.abs(np.clip(model.g_lb - Cx @ self.dx[t] - Cu @ self.du[t] - data.g, 0, np.inf)))
            l2 = np.max(np.abs(np.clip( Cx @ self.dx[t] + Cu @ self.du[t] + data.g - model.g_ub, 0, np.inf)))
            l3 =  np.max(np.abs(self.dx[t+1] - data.Fx @ self.dx[t] - data.Fu @ self.du[t] - self.gap[t]))
            KKT = max(KKT, l1, l2, l3)

        model = self.problem.terminalModel
        data = self.problem.terminalData
        Cx = data.Gx
        if model.ng != 0:
            l1 = np.max(np.abs(np.clip(model.g_lb - Cx @ self.dx[-1] - data.g, 0, np.inf)))
            l2 = np.max(np.abs(np.clip(Cx @ self.dx[-1] + data.g - model.g_ub, 0, np.inf)))
        lx = self.problem.terminalData.Lxx @ self.dx[-1] + self.problem.terminalData.Lx - self.lag_mul[-1] +  Cx.T @ self.y[-1]
        KKT =  max(KKT, l1, l2)
        KKT = max(KKT, max(abs(lx)))

        # Note that for this test to pass, the tolerance of the QP should be low.
        # assert KKT < 1e-6
        print("\n This should match the tolerance of the QP solver ", KKT)

    def KKT_check(self):
        if not self.using_qp:
            for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                self.lag_mul[t] = self.Vxx[t] @ self.dx_tilde[t] + self.Vx[t]
            self.lag_mul[-1] = self.Vxx[-1] @ self.dx_tilde[-1] + self.Vx[-1]
        self.KKT = 0
        for t, data in enumerate(self.problem.runningDatas):
            Cx, Cu = data.Gx, data.Gu
            if t==0:
                lu =  data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
                self.KKT = max(self.KKT, max(abs(lu)))
                continue
            Cx, Cu = data.Gx, data.Gu
            lx = data.Lx + data.Fx.T @ self.lag_mul[t+1] - self.lag_mul[t] + Cx.T @ self.y[t]
            lu = data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
            self.KKT = max(self.KKT, max(abs(lx)), max(abs(lu)))

        Cx = self.problem.terminalData.Gx
        lx = self.problem.terminalData.Lx - self.lag_mul[-1] +  Cx.T @ self.y[-1]
        self.KKT = max(self.KKT, max(abs(lx)))
        self.KKT = max(self.KKT, max(abs(np.array(self.gap).flatten())))
        self.KKT = max(self.KKT, self.constraint_norm)



    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0] = self.problem.x0.copy() # Initial condition guess must be x0
        
        self.gap_list = collections.deque(self.filter_size * [np.inf], maxlen=self.filter_size)
        self.cost_list = collections.deque(self.filter_size * [np.inf], maxlen=self.filter_size)
        self.constraint_list = collections.deque(self.filter_size * [np.inf], maxlen=self.filter_size)

        self.setCandidate(init_xs, init_us, False)

        alpha = None
        if (self.with_callbacks):
            headings = ["iter", "merit", "cost",  "||gaps||", "||Constraint||", "||(dx,du)||", "step", "KKT", "QP Iters"]
            
            print("{:>3} {:>7} {:>8} {:>8} {:>8} {:>11} {:>11} {:>8} {:>8}".format(*headings))
        for iter in range(maxiter):
            self.iter = iter
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

                if(self.with_callbacks):
                    print("{:>4} {:.4e} {:.4e} {:.4e} {:>4} {:.4e} {:>4} {:.4e} {:>4}".format("END", float(self.merit), self.cost, self.gap_norm, self.constraint_norm, self.x_grad_norm + self.u_grad_norm, " ---- ",  self.KKT,  self.qp_iters))
                return True
            
            
            self.gap_list.append(self.gap_norm)
            self.cost_list.append(self.cost)
            self.constraint_list.append(self.constraint_norm)
            alpha = 1.
            max_search = 10
            for k in range(max_search):
                self.tryStep(alpha)
                if self.use_filter_line_search:
                    is_worse_than_memory = False
                    count = 0
                    while count < self.filter_size and not is_worse_than_memory and count <= iter:
                        is_worse_than_memory = self.cost_list[self.filter_size - 1 - count] < self.cost_try and \
                            self.gap_list[self.filter_size - 1 - count] < self.gap_norm_try
                        count += 1
                        
                    if is_worse_than_memory == False:
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

            if(self.with_callbacks):
                print("{:>4} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4f} {:.4e} {:>4}".format(iter + 1, float(self.merit), self.cost, self.gap_norm, self.constraint_norm, self.x_grad_norm + self.u_grad_norm, alpha, self.KKT, self.qp_iters))

        if self.extra_iteration_for_last_kkt:
            self.calc(True)
            if self.using_qp:
                self.computeDirectionFullQP()
            else:
                self.computeDirection()
            if(self.BENCHMARK):
                self.check_qp_convergence()
            self.KKT_check()
            if(self.with_callbacks):
                print("{:>4} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:>4} {:.4e} {:>4}".format("END", float(self.merit), self.cost,  self.gap_norm, self.constraint_norm, self.x_grad_norm + self.u_grad_norm, " ---- ", self.KKT,  " ---- "))
            if self.KKT < self.termination_tolerance:
                return True


        return False 
    
    def check_qp_convergence(self):
        '''
        Checks the QP convergence based on the approach employed in the qpsolvers pkg
        '''
        # If we are benchmarking, setup the problem using QPSolvers (S. Caron)
        # in order to get the unified convergence metrics (prim_res, dual_res, dual_gap)
        # We follow the convention of QPSolvers package, i.e.
        # min_x (1/2) x^T P x + q^T x
        #   s.t. Ax  = b
        #        Gx <= h
        # see here https://github.com/qpsolvers/qpsolvers/blob/main/qpsolvers/problem.py
        P, q, A, B, C, u, l = self.create_full_qp()
        P_qp = P ; q_qp = q
        A_qp = A ; b_qp = B
        G_qp = np.vstack([C, -C])
        h_qp = np.hstack([u, -l]) 
        prob_qp = Problem(P_qp, q_qp, G_qp, h_qp, A_qp, b_qp) # no box constraints
        sol_qp = Solution(prob_qp)

        if(self.using_qp):
            if(self.method == 'OSQP'):
                # sol_qp.found = self.osqp_sol_found
                # Solution
                sol_qp.x = self.osqp_results.x
                m = C.shape[0] if G_qp is not None else 0 ; meq = A_qp.shape[0] if A_qp is not None else 0
                # Lagrange multipliers for equality constraint Ax = b
                sol_qp.y = self.osqp_results.y[:meq] if A_qp is not None else np.empty((0,))
                # Lagrange multipliers for inequality constraint Gx <= h ( a.k.a. "Cx <= ub" and "-Cx <= -lb" )
                z = self.osqp_results.y[meq:meq+m] if G_qp is not None else np.empty((0,))
                zp = np.maximum(z, np.zeros_like(z)) # see OSQP paper Eq. (9)
                zm = np.minimum(z, np.zeros_like(z)) # see OSQP paper Eq. (9)
                sol_qp.z = np.hstack([zp, zm])
                # No box constraints in the present formulation
                sol_qp.z_box = np.empty((0,))
            if(self.method == 'HPIPM'):
                pass
        else:
            sol_qp.x
        # Print convergence metrics
        self.is_optimal      = sol_qp.is_optimal(self.eps_abs)
        self.primal_residual = sol_qp.primal_residual()
        self.dual_residual   = sol_qp.dual_residual()
        self.duality_gap     = sol_qp.duality_gap()
        print(f"- Solution is{'' if sol_qp.is_optimal(1e-3) else ' NOT'} optimal")
        print(f"- Primal residual: {sol_qp.primal_residual():.1e}")
        print(f"- Dual residual: {sol_qp.dual_residual():.1e}")
        print(f"- Duality gap: {sol_qp.duality_gap():.1e}")
            # else:
                    # self.is_optimal      = False
                    # self.primal_residual = np.inf
                    # self.dual_residual   = np.inf
                    # self.duality_gap     = np.inf


    def create_full_qp(self):
        '''
        Creates the full QP (needed for benchmarks)
        '''
        self.n_vars  = self.problem.T*(self.ndx + self.nu)
        P = np.zeros((self.problem.T*(self.ndx + self.nu), self.problem.T*(self.ndx + self.nu)))
        q = np.zeros(self.problem.T*(self.ndx + self.nu))
        Asize = self.problem.T*(self.ndx + self.nu)
        A = np.zeros((self.problem.T*self.ndx, Asize))
        B = np.zeros(self.problem.T*self.ndx)
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            index_u = self.problem.T*self.ndx + t * self.nu
            if t>=1:
                index_x = (t-1) * self.ndx
                P[index_x:index_x+self.ndx, index_x:index_x+self.ndx] = data.Lxx.copy()
                P[index_x:index_x+self.ndx, index_u:index_u+self.nu] = data.Lxu.copy()
                P[index_u:index_u+self.nu, index_x:index_x+self.ndx] = data.Lxu.T.copy()
                q[index_x:index_x+self.ndx] = data.Lx.copy()
            P[index_u:index_u+self.nu, index_u:index_u+self.nu] = data.Luu.copy()
            q[index_u:index_u+self.nu] = data.Lu.copy()
            A[t * self.ndx: (t+1) * self.ndx, index_u:index_u+self.nu] = - data.Fu.copy() 
            A[t * self.ndx: (t+1) * self.ndx, t * self.ndx: (t+1) * self.ndx] = np.eye(self.ndx)
            if t >=1:
                A[t * self.ndx: (t+1) * self.ndx, (t-1) * self.ndx: t * self.ndx] = - data.Fx.copy()
            B[t * self.ndx: (t+1) * self.ndx] = self.gap[t].copy()
        P[(self.problem.T-1)*self.ndx:self.problem.T*self.ndx, self.problem.T*self.ndx-self.ndx:self.problem.T*self.ndx] = self.problem.terminalData.Lxx.copy()
        q[(self.problem.T-1)*self.ndx:self.problem.T*self.ndx] = self.problem.terminalData.Lx.copy()
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
            C[nin_count: nin_count + model.ng, index_x+t*self.nu: index_x+(t+1)*self.nu] = data.Gu
            nin_count += model.ng
        model = self.problem.terminalModel
        data = self.problem.terminalData
        if model.ng != 0:
            l[nin_count: nin_count + model.ng] = model.g_lb - data.g
            u[nin_count: nin_count + model.ng] = model.g_ub - data.g
            C[nin_count: nin_count + model.ng, (self.problem.T-1)*self.ndx: self.problem.T*self.ndx] = data.Gx
            nin_count += model.ng
        return P, q, A, B, C, u, l