## This is the implementation of the constrained sparse QP Solver
## Author : Avadesh Meduri and Armand Jordana
## Date : 8/03/2023

import numpy as np
from crocoddyl import SolverAbstract
import scipy.linalg as scl
import eigenpy
import time

LINE_WIDTH = 100 


pp = lambda s : np.format_float_scientific(s, exp_digits=2, precision =4)

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class StagewiseQP(SolverAbstract):
    def __init__(self, shootingProblem, constraintModel, verboseQP = False):
        SolverAbstract.__init__(self, shootingProblem)        
        self.constraintModel = constraintModel

        self.reset_params()
        
        self.allocateQPData()
        self.allocateData()

        self.max_iters = 3000

        self.eps_abs = 1e-4
        self.eps_rel = 1e-4
        self.adaptive_rho_tolerance = 5
        self.rho_update_interval = 25
        self.regMin = 1e-6
        self.alpha = 1.6

        self.verboseQP = verboseQP

        self.OSQP_update = True

        if self.verboseQP:
            print("USING StagewiseQP")

    def reset_params(self):
        
        self.sigma_sparse = 1e-6
        self.rho_sparse= 1e-1
        self.rho_min = 1e-6
        self.rho_max = 1e6

        self.z = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]
        self.y = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]


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

        for t, (cmodel, cdata, data) in enumerate(zip(self.constraintModel[:-1], self.constraintData[:-1], self.problem.runningDatas)):
            cmodel.calc(cdata, data, self.xs[t], self.us[t])
            cmodel.calcDiff(cdata, data, self.xs[t], self.us[t])
            self.constraint_norm += np.linalg.norm(np.clip(cmodel.lb - cdata.c, 0, np.inf), 1) 
            self.constraint_norm += np.linalg.norm(np.clip(cdata.c - cmodel.ub, 0, np.inf), 1)

        cmodel, cdata = self.constraintModel[-1], self.constraintData[-1]
        cmodel.calc(cdata, self.problem.terminalData, self.xs[-1], np.zeros(7))
        cmodel.calcDiff(cdata, self.problem.terminalData, self.xs[-1], np.zeros(7))

        self.constraint_norm += np.linalg.norm(np.clip(cmodel.lb - cdata.c, 0, np.inf), 1) 
        self.constraint_norm += np.linalg.norm(np.clip(cdata.c - cmodel.ub, 0, np.inf), 1)

        for t, (model, data) in enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            # model.calc(data, self.xs[t], self.us[t])  
            self.gap[t] = model.state.diff(self.xs[t+1].copy(), data.xnext.copy())
            self.cost += data.cost


        self.gap_norm = sum(np.linalg.norm(self.gap.copy(), 1, axis = 1))
        self.cost += self.problem.terminalData.cost 
        self.gap = self.gap.copy()

    def computeDirection(self, KKT=True):
        self.calc(True)
        if KKT:
            self.KKT_check()

        self.reset_params()
        self.backwardPass_without_constraints()
        self.computeUpdates()
        self.update_lagrangian_parameters_infinity(False)
        
        for iter in range(1, self.max_iters+1):
            if (iter) % self.rho_update_interval == 1 or iter == 1:
                self.backwardPass() 
 
            else:
                # self.backwardPass()  
                self.backwardPass_without_rho_update()
            
            self.computeUpdates()
            self.update_lagrangian_parameters_infinity(True)
            self.update_rho_sparse(iter)

            if self.norm_primal <= self.eps_abs + self.eps_rel*self.norm_primal_rel and\
                    self.norm_dual <= self.eps_abs + self.eps_rel*self.norm_dual_rel:
                        if self.verboseQP:
                            print("Iters", iter, "res-primal", pp(self.norm_primal), "res-dual", pp(self.norm_dual)\
                                , "optimal rho estimate", pp(self.rho_estimate_sparse), "rho", pp(self.rho_sparse)) 
                            print("StagewiseQP converged", "\n")
                        break

            if (iter) % self.rho_update_interval == 0 and iter > 1:
                if self.verboseQP:
                    print("Iters", iter, "res-primal", pp(self.norm_primal), "res-dual", pp(self.norm_dual)\
                    , "optimal rho estimate", pp(self.rho_estimate_sparse), "rho", pp(self.rho_sparse)) 
        if self.verboseQP:
            print("\n")
        
        self.QP_iter = iter

    def update_rho_sparse(self, iter):
        if self.OSQP_update:
            scale = (self.norm_primal * self.norm_dual_rel)/(self.norm_dual * self.norm_primal_rel)
        else:
            scale = (self.kkt_primal)/(self.kkt_dual)
        scale = np.sqrt(scale)
        self.scale_sparse = scale
        self.rho_estimate_sparse = scale * self.rho_sparse
        self.rho_estimate_sparse = min(max(self.rho_estimate_sparse, self.rho_min), self.rho_max) 


        if (iter) % self.rho_update_interval == 0 and iter > 1:
            if self.rho_estimate_sparse > self.rho_sparse* self.adaptive_rho_tolerance or\
                self.rho_estimate_sparse < self.rho_sparse/ self.adaptive_rho_tolerance :
                self.rho_sparse= self.rho_estimate_sparse
                
                for t, cmodel in enumerate(self.constraintModel):   
                    if t == self.problem.T:
                        scaler = 1
                    else:
                        scaler = 1
                    
                    for k in range(cmodel.nc):  
                        if cmodel.lb[k] == -np.inf and cmodel.ub[k] == np.inf:
                            self.rho_vec[t][k] = self.rho_min 
                        elif abs(cmodel.lb[k] - cmodel.ub[k]) < 1e-3:
                            self.rho_vec[t][k] = scaler *1e3 * self.rho_sparse
                        elif cmodel.lb[k] != cmodel.ub[k]:
                            self.rho_vec[t][k] = scaler * self.rho_sparse

    def update_lagrangian_parameters_infinity(self, update_y):

        self.norm_primal = -np.inf
        self.norm_dual = -np.inf
        self.kkt_primal = -np.inf
        self.kkt_dual = -np.inf

        self.norm_primal_rel, self.norm_dual_rel = [-np.inf,-np.inf], -np.inf
        
        for t, (cmodel, cdata) in enumerate(zip(self.constraintModel[:-1], self.constraintData[:-1])):
            if cmodel.nc == 0:
                self.dx[t] = self.dx_tilde[t].copy()
                self.du[t] = self.du_tilde[t].copy()
                continue

            Cx, Cu = cdata.Cx, cdata.Cu

            z_k = self.z[t].copy()
            
            Cdx_Cdu = Cx @ self.dx_tilde[t].copy() + Cu @ self.du_tilde[t].copy()

            self.dz_relaxed[t] = self.alpha * (Cdx_Cdu) + (1 - self.alpha)*self.z[t]

            self.z[t] = np.clip(self.dz_relaxed[t] + np.divide(self.y[t], self.rho_vec[t]), cmodel.lb - cdata.c, cmodel.ub - cdata.c)
            if update_y:
                self.y[t] += np.multiply(self.rho_vec[t], (self.dz_relaxed[t] - self.z[t])) 

            self.dx[t] = self.dx_tilde[t].copy()
            self.du[t] = self.du_tilde[t].copy()

            # OSQP
            dual_vecx = Cx.T @  np.multiply(self.rho_vec[t], (self.z[t] - z_k)) 
            dual_vecu = Cu.T @  np.multiply(self.rho_vec[t], (self.z[t] - z_k)) 
            self.norm_dual = max(self.norm_dual, max(abs(dual_vecx)), max(abs(dual_vecu)))
            self.norm_primal = max(self.norm_primal, max(abs(Cdx_Cdu - self.z[t])))

            # KKT
            data = self.problem.runningDatas[t]
            dual_vecx = data.Lxx @ self.dx[t] + data.Lxu @ self.du[t] + data.Lx + data.Fx.T @ self.lag_mul[t+1] - self.lag_mul[t] + Cx.T @ self.y[t]
            dual_vecu = data.Luu @ self.du[t] + data.Lxu.T @ self.dx[t] + data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
            self.kkt_dual = max(self.kkt_dual, max(abs(dual_vecx)), max(abs(dual_vecu)))
            l1 = np.max(np.abs(np.clip(cmodel.lb - Cx @ self.dx[t] - Cu @ self.du[t] - cdata.c, 0, np.inf)))
            l2 = np.max(np.abs(np.clip( Cx @ self.dx[t] + Cu @ self.du[t] + cdata.c - cmodel.ub, 0, np.inf)))
            self.kkt_primal = max(self.kkt_primal, l1, l2)

            self.norm_primal_rel[0] = max(self.norm_primal_rel[0], max(abs(Cdx_Cdu)))
            self.norm_primal_rel[1] = max(self.norm_primal_rel[1], max(abs(self.z[t])))
            self.norm_dual_rel = max(self.norm_dual_rel, max(abs(Cx.T@self.y[t])), max(abs(Cu.T@self.y[t])))

        self.dx[-1] = self.dx_tilde[-1].copy()
        if self.constraintModel[-1].nc != 0:
            cmodel = self.constraintModel[-1]
            cdata = self.constraintData[-1]
            Cx = cdata.Cx

            z_k = self.z[-1].copy()

            Cdx = Cx @ self.dx_tilde[-1].copy()

            self.dz_relaxed[-1] = self.alpha * Cdx + (1 - self.alpha)*self.z[-1]
            self.z[-1] = np.clip(self.dz_relaxed[-1] + np.divide(self.y[-1], self.rho_vec[-1]), cmodel.lb - cdata.c, cmodel.ub - cdata.c)
            # print(cdata.c)
            # print(self.dz_relaxed[-1])
            
            if update_y:
                self.y[-1] += np.multiply(self.rho_vec[-1], (self.dz_relaxed[-1] - self.z[-1])) 

            self.dx[-1] = self.dx_tilde[-1].copy()

            # OSQP
            self.norm_primal = max(self.norm_primal, max(abs(Cdx - self.z[-1])))
            dual_vec = Cx.T@np.multiply(self.rho_vec[-1], (self.z[-1] - z_k))
            self.norm_dual = max(self.norm_dual, max(abs(dual_vec)))

            # KKT
            dual_vec = self.problem.terminalData.Lxx @ self.dx[-1] + self.problem.terminalData.Lx - self.lag_mul[-1] +  Cx.T @ self.y[-1]
            self.kkt_dual = max(self.kkt_dual, max(abs(dual_vec)))
            l1 = np.max(np.abs(np.clip(cmodel.lb - Cx @ self.dx[-1] - cdata.c, 0, np.inf)))
            l2 = np.max(np.abs(np.clip( Cx @ self.dx[-1] + cdata.c - cmodel.ub, 0, np.inf)))
            self.kkt_primal = max(self.kkt_primal, l1, l2)


            self.norm_primal_rel[0] = max(self.norm_primal_rel[0], max(abs(Cx@self.dx[-1])))
            self.norm_primal_rel[1] = max(self.norm_primal_rel[1], max(abs(self.z[-1])))
            self.norm_dual_rel = max(self.norm_dual_rel, max(abs(Cx.T@self.y[-1])))
        self.norm_primal_rel = max(self.norm_primal_rel)


    def acceptStep(self, alpha):
        
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.xs_try[t] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t] = self.us[t] + alpha*self.du[t]    
        self.xs_try[-1] = model.state.integrate(self.xs[-1], alpha*self.dx[-1]) ## terminal state update

        self.setCandidate(self.xs_try, self.us_try, False)

    def computeUpdates(self): 
        """ computes step updates dx and du """
        self.expected_decrease = 0
        assert np.linalg.norm(self.dx[0]) < 1e-6
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                self.lag_mul[t] = self.S[t] @ self.dx_tilde[t] + self.s[t]
                self.du_tilde[t][:] = self.L[t].dot(self.dx_tilde[t]) + self.l[t] 
                A = data.Fx.copy()
                B = data.Fu.copy()      
                if len(B.shape) == 1:
                    bl = B.dot(self.l[t][0])
                    BL = B.reshape(B.shape[0], 1)@self.L[t]
                else: 
                    bl = B @ self.l[t]
                    BL = B@self.L[t]
                self.dx_tilde[t+1] = (A + BL)@self.dx_tilde[t] + bl + self.gap[t].copy()  

        self.lag_mul[-1] = self.S[-1] @ self.dx_tilde[-1] + self.s[-1]

        self.x_grad_norm = np.linalg.norm(self.dx_tilde)/(self.problem.T+1)
        self.u_grad_norm = np.linalg.norm(self.du_tilde)/self.problem.T

    def backwardPass(self): 
        rho_mat = self.rho_vec[-1]*np.eye(len(self.rho_vec[-1]))
        self.S[-1][:,:] = self.problem.terminalData.Lxx.copy() + self.sigma_sparse*np.eye(self.problem.terminalModel.state.nx) 
        self.s[-1][:] = self.problem.terminalData.Lx.copy() - self.sigma_sparse * self.dx[-1]
        if self.constraintModel[-1].nc != 0:
            Cx = self.constraintData[-1].Cx
            self.S[-1][:,:] +=  Cx.T @ rho_mat @ Cx 
            self.s[-1][:] +=  Cx.T @ ( self.y[-1] - rho_mat @ self.z[-1])[:]  

        for t, (model, data, cdata) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas, self.constraintData[:-1])):
            rho_mat = self.rho_vec[t]*np.eye(len(self.rho_vec[t]))
            q = data.Lx.copy() - self.sigma_sparse * self.dx[t] 
            Q = data.Lxx.copy() + self.sigma_sparse*np.eye(model.state.nx)
            r = data.Lu.copy() - self.sigma_sparse * self.du[t]
            R = data.Luu.copy() + self.sigma_sparse*np.eye(model.nu)
            P = data.Lxu.T.copy()

            if self.constraintModel[t].nc != 0:
                Cx, Cu = cdata.Cx, cdata.Cu
                r += Cu.T @ (self.y[t] - rho_mat @self.z[t])[:]
                R += Cu.T @ rho_mat @ Cu

                if t > 0:
                    q += Cx.T@(self.y[t] - rho_mat@self.z[t])[:]
                    Q += Cx.T @ rho_mat @ Cx
                    P += Cu.T @ rho_mat @ Cx

            A = data.Fx.copy()    
            B = data.Fu.copy() 
            h = r + B.T@(self.s[t+1] + self.S[t+1]@self.gap[t].copy())
            self.G[t] = P + B.T@self.S[t+1]@A
            if len(self.G[t].shape) == 1:
                self.G[t] = np.resize(self.G[t],(1,self.G[t].shape[0]))
            
            self.H[t] = R + B.T@self.S[t+1]@B

            self.H_llt[t] = eigenpy.LLT(self.H[t])
            self.L[t][:,:] = -1* self.H_llt[t].solve(self.G[t])
            self.l[t][:] = -1*self.H_llt[t].solve(h)
        

            self.S[t] = Q + A.T @ (self.S[t+1])@A - self.L[t].T@self.H[t]@self.L[t] 
            self.s[t] = q + A.T @ (self.S[t+1] @ self.gap[t].copy() + self.s[t+1]) + \
                            self.G[t].T@self.l[t][:]+ self.L[t].T@(h + self.H[t]@self.l[t][:])

    def backwardPass_without_constraints(self): 
        self.S[-1][:,:] = self.problem.terminalData.Lxx.copy()
        self.s[-1][:] = self.problem.terminalData.Lx.copy()
        
        for t, (model, data, cdata) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas, self.constraintData[:-1])):
            rho_mat = self.rho_vec[t]*np.eye(len(self.rho_vec[t]))
            q = data.Lx.copy() 
            Q = data.Lxx.copy()
            r = data.Lu.copy()
            R = data.Luu.copy()
            P = data.Lxu.T.copy()

            A = data.Fx.copy()    
            B = data.Fu.copy() 
            h = r + B.T@(self.s[t+1] + self.S[t+1]@self.gap[t].copy())
            self.G[t] = P + B.T@self.S[t+1]@A
            if len(self.G[t].shape) == 1:
                self.G[t] = np.resize(self.G[t],(1,self.G[t].shape[0]))
            
            self.H[t] = R + B.T@self.S[t+1]@B

            self.H_llt[t] = eigenpy.LLT(self.H[t])
            self.L[t][:,:] = -1* self.H_llt[t].solve(self.G[t])
            self.l[t][:] = -1*self.H_llt[t].solve(h)
        

            self.S[t] = Q + A.T @ (self.S[t+1])@A - self.L[t].T@self.H[t]@self.L[t] 
            self.s[t] = q + A.T @ (self.S[t+1] @ self.gap[t].copy() + self.s[t+1]) + \
                            self.G[t].T@self.l[t][:]+ self.L[t].T@(h + self.H[t]@self.l[t][:])


    def backwardPass_without_rho_update(self): 
        self.s[-1][:] = self.problem.terminalData.Lx.copy() - self.sigma_sparse * self.dx[-1]
        if self.constraintModel[-1].nc != 0:
            rho_mat = self.rho_vec[-1]*np.eye(len(self.rho_vec[-1]))
            Cx = self.constraintData[-1].Cx
            self.s[-1][:] +=  Cx.T @ ( self.y[-1] - rho_mat @ self.z[-1])[:]  

        for t, (model, data, cdata) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas, self.constraintData[:-1])):
            rho_mat = self.rho_vec[t]*np.eye(len(self.rho_vec[t]))
            q = data.Lx.copy() - self.sigma_sparse * self.dx[t] 
            r = data.Lu.copy() - self.sigma_sparse * self.du[t]

            if self.constraintModel[t].nc != 0:
                Cx, Cu = cdata.Cx, cdata.Cu
                r += Cu.T @ (self.y[t] - rho_mat @self.z[t])[:]

                if t > 0:
                    q += Cx.T@(self.y[t] - rho_mat@self.z[t])[:]

            A = data.Fx.copy()    
            B = data.Fu.copy() 
            h = r + B.T@(self.s[t+1] + self.S[t+1]@self.gap[t].copy())
            
            self.l[t][:] = -1*self.H_llt[t].solve(h)
        
            self.s[t] = q + A.T @ (self.S[t+1] @ self.gap[t].copy() + self.s[t+1]) + \
                            self.G[t].T@self.l[t][:]+ self.L[t].T@(h + self.H[t]@self.l[t][:])
            

    def solve(self, init_xs=None, init_us=None, maxiter=1000, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        self.max_iters = maxiter
        if init_xs is None or len(init_xs) < 1:
            init_xs = [self.problem.x0.copy() for m in self.models()] 
        if init_us is None or len(init_us) < 1:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 

        init_xs[0][:] = self.problem.x0.copy() # Initial condition guess must be x0
        self.setCandidate(init_xs, init_us, False)
        self.computeDirection(KKT=False)

        self.acceptStep(alpha = 1.0)
        # self.reset_params()
        
    def allocateQPData(self):

        self.z = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]
        self.y = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]

        self.z_test = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]
        self.y_test = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]

        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.dx_tilde = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du_tilde = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        #
        self.dx_test = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du_test = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        # 
        self.lag_mul = [np.zeros(m.state.ndx) for m  in self.models()] 
        self.dz_relaxed = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]

        self.rho_vec = [np.zeros(cmodel.nc) for cmodel in self.constraintModel]
        self.rho_estimate_sparse = 0.0
        self.rho_sparse = min(max(self.rho_sparse, self.rho_min), self.rho_max) 
        for t, cmodel in enumerate(self.constraintModel):   
            if t == self.problem.T:
                scaler = 1
            else:
                scaler = 1
            for k in range(cmodel.nc):  
                if cmodel.lb[k] == -np.inf and cmodel.ub[k] == np.inf:
                    self.rho_vec[t][k] = self.rho_min 
                elif abs(cmodel.lb[k] - cmodel.ub[k]) < 1e-3:
                    self.rho_vec[t][k] = scaler * 1e3 * self.rho_sparse
                elif cmodel.lb[k] != cmodel.ub[k]:
                    self.rho_vec[t][k] = scaler * self.rho_sparse


    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        # 
        #  
        self.constraintData = [cmodel.createData() for cmodel in self.constraintModel]
        #
        self.S = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]   
        self.s = [np.zeros(m.state.ndx) for m in self.models()]   
        self.L = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.l = [np.zeros([m.nu]) for m in self.problem.runningModels]
        
        self.H = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]   
        self.G = [np.zeros([m.state.nx, m.state.nx]) for m in self.problem.runningModels]   
        self.H_llt = [np.array([0]) for m in self.problem.runningModels]
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
        self.nu = self.problem.runningModels[0].nu


        self.QP_iter = 0