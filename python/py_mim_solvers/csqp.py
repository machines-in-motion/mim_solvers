## Implementation of the sequential constrained lqr
## Author : Avadesh Meduri and Armand Jordana
## Date : 9/3/2022

import crocoddyl
from crocoddyl import SolverAbstract
import numpy as np
import scipy.linalg as scl

pp = lambda s : np.format_float_scientific(s, exp_digits=2, precision =4)

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class CSSQP(StagewiseQP):

    def __init__(self, shootingProblem, use_filter_ls=True, verboseQP = False, verbose = False):
        self.verbose = verbose
        StagewiseQP.__init__(self, shootingProblem)

        self.mu1 = 1e1
        self.mu2 = 1e1
        self.termination_tol = 1e-8
        self.use_filter_ls = use_filter_ls

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


        for t, (cmodel, cdata, data) in enumerate(zip(self.constraintModel[:-1], self.constraintData[:-1], self.problem.runningDatas)):
            cmodel.calc(cdata, data, self.xs_try[t], self.us_try[t])

            self.constraint_norm_try += np.linalg.norm(np.clip(cmodel.lb - cdata.c, 0, np.inf), 1) 
            self.constraint_norm_try += np.linalg.norm(np.clip(cdata.c - cmodel.ub, 0, np.inf), 1)

        cmodel, cdata = self.constraintModel[-1], self.constraintData[-1]
        cmodel.calc(cdata, self.problem.terminalData, self.xs_try[-1], np.zeros(4))

        self.constraint_norm_try += np.linalg.norm(np.clip(cmodel.lb - cdata.c, 0, np.inf), 1) 
        self.constraint_norm_try += np.linalg.norm(np.clip(cdata.c - cmodel.ub, 0, np.inf), 1)


        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])  
            self.gap_try[t] = model.state.diff(self.xs_try[t+1], data.xnext) #gaps
            self.cost_try += data.cost

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        self.cost_try += self.problem.terminalData.cost

        self.gap_norm_try = sum(np.linalg.norm(self.gap_try, 1, axis = 1))

        self.merit_try =  self.cost_try + self.mu1 * self.gap_norm_try + self.mu2 * self.constraint_norm_try

    def LQ_problem_KKT_check(self):
        KKT = 0
        for t, (cmodel, cdata, data) in enumerate(zip(self.constraintModel[:-1], self.constraintData[:-1], self.problem.runningDatas)):
            Cx, Cu = cdata.Cx, cdata.Cu
            if t == 0:
                lu = data.Luu @ self.du[t] + data.Lxu.T @ self.dx[t] + data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
                KKT = max(KKT, max(abs(lu)))
                continue

            lx = data.Lxx @ self.dx[t] + data.Lxu @ self.du[t] + data.Lx + data.Fx.T @ self.lag_mul[t+1] - self.lag_mul[t] + Cx.T @ self.y[t]
            lu = data.Luu @ self.du[t] + data.Lxu.T @ self.dx[t] + data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
            KKT = max(KKT, max(abs(lx)), max(abs(lu)))

            l1 = np.max(np.abs(np.clip(cmodel.lmin - Cx @ self.dx[t] - Cu @ self.du[t] - cdata.c, 0, np.inf)))
            l2 = np.max(np.abs(np.clip( Cx @ self.dx[t] + Cu @ self.du[t] + cdata.c - cmodel.lmax, 0, np.inf)))
            l3 =  np.max(np.abs(self.dx[t+1] - data.Fx @ self.dx[t] - data.Fu @ self.du[t] - self.gap[t]))
            KKT = max(KKT, l1, l2, l3)

        cmodel = self.constraintModel[-1]
        cdata = self.constraintData[-1]
        Cx = cdata.Cx

        l1 = np.max(np.abs(np.clip(cmodel.lmin - Cx @ self.dx[-1] - cdata.c, 0, np.inf)))
        l2 = np.max(np.abs(np.clip(Cx @ self.dx[-1] + cdata.c- cmodel.lmax, 0, np.inf)))
        lx = self.problem.terminalData.Lxx @ self.dx[-1] + self.problem.terminalData.Lx - self.lag_mul[-1] +  Cx.T @ self.y[-1]
        KKT =  max(KKT, l1, l2)
        KKT = max(KKT, max(abs(lx)))

        # Note that for this test to pass, the tolerance of the QP should be low.
        # assert KKT < 1e-6
        print("\n This should match the tolerance of the QP solver ", KKT)

    def KKT_check(self):
        # print(self.lag_mul)
        # print(self.y)
        self.KKT = 0
        for t, (cdata, data) in enumerate(zip(self.constraintData[:-1], self.problem.runningDatas)):
            Cx, Cu = cdata.Cx, cdata.Cu
            if t==0:
                lu =  data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
                self.KKT = max(self.KKT, max(abs(lu)))
                continue
            Cx, Cu = cdata.Cx, cdata.Cu
            lx = data.Lx + data.Fx.T @ self.lag_mul[t+1] - self.lag_mul[t] + Cx.T @ self.y[t]
            lu = data.Lu + data.Fu.T @ self.lag_mul[t+1] + Cu.T @ self.y[t]
            self.KKT = max(self.KKT, max(abs(lx)), max(abs(lu)))

        Cx = self.constraintData[-1].Cx
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
        self.setCandidate(init_xs, init_us, False)
        if self.verbose:
            header = "{: >5} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >10}".format(*["iter", "KKT norm", "merit", "cost", "gap norm", "constraint norm", "QP iter ", "dx norm", "du norm", "alpha"])

        cost_list = []
        gap_list = []
        constraint_list = []
        alpha = None
        for i in range(maxiter):
            if self.verbose and i % 40 == 0:
                print("\n", header)

            if self.using_qp:
                self.computeDirectionFullQP()
            else:
                self.computeDirection()
                
            # self.LQ_problem_KKT_check()

            cost_list.append(self.cost)
            gap_list.append(self.gap_norm)
            constraint_list.append(self.constraint_norm)


            self.merit =  self.cost + self.mu1 * self.gap_norm + self.mu2 * self.constraint_norm
            if self.verbose:
                print("{: >5} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >10}".format(*[i, pp(self.KKT), pp(self.merit), pp(self.cost), pp(self.gap_norm), pp(self.constraint_norm), self.QP_iter, pp(self.x_grad_norm), pp(self.u_grad_norm), str(alpha)]))

            alpha = 1.
            self.tryStep(alpha)
            max_search = 20
            for k in range(max_search):
                if k == max_search - 1:
                    print("No improvement")
                    return False

                # if self.merit < self.merit_try:
                if self.use_heuristic_ls:
                    filter_list = [constraint < self.constraint_norm_try and gap < self.gap_norm_try and cost < self.cost_try for (constraint, gap, cost) in zip(constraint_list, gap_list, cost_list)]
                    # if np.array(filter_list).any():
                    if self.cost < self.cost_try and self.gap_norm < self.gap_norm_try and self.constraint_norm < self.constraint_norm_try:
                        alpha *= 0.5
                        self.tryStep(alpha)
                    else:
                        self.setCandidate(self.xs_try, self.us_try, False)
                        break
                else:
                    if self.merit < self.merit_try:
                        alpha *= 0.5
                        self.tryStep(alpha)
                    else:
                        self.setCandidate(self.xs_try, self.us_try, False)
                        break

            if self.KKT < self.termination_tol:
                if self.verbose:
                    print("Converged")
                break
        if self.verbose:
            self.calc()
            # self.KKT_check()
            print("{: >5} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >14} {: >10}".format(*["Final", pp(self.KKT), pp(self.merit), pp(self.cost), pp(self.gap_norm),  pp(self.constraint_norm), " ", " ", " ", " "]))




    