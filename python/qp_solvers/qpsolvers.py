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

class QPSolvers(SolverAbstract, CustomOSQP, StagewiseQPKKT):

    def __init__(self, shootingProblem, method, verboseQP = True):
        
        SolverAbstract.__init__(self, shootingProblem)        

        assert method == "ProxQP" or method=="OSQP"\
              or method=="CustomOSQP" or method =="StagewiseQPKKT" 
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
            C[nin_count: nin_count + model.ng, index_x+t*self.nu: index_x+(t+1)*self.nu] = data.Gu
            nin_count += model.ng

            # import pdb; pdb.set_trace()
#
        model = self.problem.terminalModel
        data = self.problem.terminalData
        if model.ng != 0:
            l[nin_count: nin_count + model.ng] = model.g_lb - data.g
            u[nin_count: nin_count + model.ng] = model.g_ub - data.g
            C[nin_count: nin_count + model.ng, (self.problem.T-1)*self.ndx: self.problem.T*self.ndx] = data.Gx
            nin_count += model.ng


        if self.method == "ProxQP":
            qp = proxsuite.proxqp.sparse.QP(n, self.n_eq, self.n_in)
            qp.settings.eps_abs = self.eps_abs
            qp.settings.eps_abs = self.eps_rel
            qp.settings.max_iter = 100
            qp.settings.max_iter_in = 100
            qp.init(P, q, A, B, C, l, u)      
            qp.solve()
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
            prob = osqp.OSQP()
            prob.setup(P, q, Aosqp, losqp, uosqp, warm_start=False, scaling=False,  max_iter = self.max_qp_iters, \
                            adaptive_rho=True, verbose = self.verboseQP, eps_rel=self.eps_rel, eps_abs=self.eps_abs)     
            # import pdb; pdb.set_trace()
            prob.eps_abs = self.eps_abs
            prob.eps_rel = self.eps_rel
  
            tmp = prob.solve()
            res = tmp.x
            self.y_k = tmp.y
            self.qp_iters = tmp.info.iter
            
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

        if self.method == "CustomOSQP" or self.method == "OSQP":
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