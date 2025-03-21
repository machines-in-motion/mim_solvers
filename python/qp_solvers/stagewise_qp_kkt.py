## Custom osqp implementation
## Author : Avadesh Meduri
## Date : 31/03/2023

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

np.set_printoptions(precision=2)
# np.format_float_scientific(self.r_prim, exp_digits=2, precision =2)


def pp(s):
    return np.format_float_scientific(s, exp_digits=2, precision=4)


class StagewiseQPKKT:
    def __init__(self):
        self.sigma_boyd = 1e-6
        self.rho_boyd = 1e-1
        self.rho_min = 1e-6
        self.rho_max = 1e3
        self.alpha_boyd = 1.6

        self.adaptive_rho_tolerance = 5
        self.rho_update_interval = 25

    def solve_linear_system_boyd(self):
        rho_mat = self.rho_vec_boyd * np.eye(len(self.rho_vec_boyd))
        A_block_leftcol = sparse.vstack(
            [
                self.P
                + self.A_in.T @ rho_mat @ self.A_in
                + self.sigma_boyd * np.eye(self.n_vars),
                self.A_eq,
            ]
        )
        A_block_rightcol = sparse.vstack(
            [self.A_eq.T, np.zeros((self.n_eq, self.n_eq))]
        )
        A_block = sparse.hstack([A_block_leftcol, A_block_rightcol])
        A_block = sparse.csr_matrix(A_block)

        tmp = self.A_in.T @ (
            np.multiply(
                self.rho_vec_boyd, self.z_k - np.divide(self.y_k, self.rho_vec_boyd)
            )
        )
        b_block = np.hstack((tmp - self.q + self.sigma_boyd * self.x_k, self.b))

        xv_k_1 = spsolve(A_block, b_block)
        self.xtilde_k_1, self.v_k_1 = xv_k_1[: self.n_vars], xv_k_1[self.n_vars :]

    def update_lagrangian_params_boyd(self):
        self.x_k_1 = (
            self.alpha_boyd * self.A_in @ self.xtilde_k_1
            + (1 - self.alpha_boyd) * self.z_k
        )  # relaxed update
        self.z_k_1 = np.clip(
            self.x_k_1 + np.divide(self.y_k, self.rho_vec_boyd), self.lboyd, self.uboyd
        )
        self.y_k_1 = self.y_k + np.multiply(
            self.rho_vec_boyd, (self.x_k_1 - self.z_k_1)
        )

        dual_vec = self.A_in.T @ np.multiply(self.rho_vec_boyd, (self.z_k_1 - self.z_k))
        self.r_dual = max(abs(dual_vec))
        self.x_k, self.z_k, self.y_k = (
            self.xtilde_k_1.copy(),
            self.z_k_1.copy(),
            self.y_k_1.copy(),
        )

        if self.A_in.shape[0] == 0:
            tmp = 0
            self.eps_rel_prim = 1e-10
            self.eps_rel_dual = 1e-10
        else:
            tmp = max(abs(self.A_in @ self.x_k - self.z_k))
            self.eps_rel_prim = max(abs(np.hstack((self.A_in @ self.x_k, self.z_k))))
            self.eps_rel_dual = max(abs(self.A_in.T @ self.y_k))

        self.r_prim = max(tmp, max(abs(self.A_eq @ self.x_k - self.b)))

        ## This is the OSQP dual computation
        # self.r_dual = max(abs(self.P @ self.x_k + self.q + self.A_in.T @ self.y_k + self.A_eq.T @ self.v_k_1))

        # tmp = max(abs(self.q))
        # tmp2 = max(abs(self.A_in.T @ self.y_k))
        # tmp3 = max(abs(self.P @ self.x_k))
        # tmp4 = max(abs(self.A_eq.T @ self.v_k_1))
        # self.eps_rel_dual = max(tmp, tmp2)
        # self.eps_rel_dual = max(self.eps_rel_dual, tmp3)
        # self.eps_rel_dual = max(self.eps_rel_dual, tmp4)

    def update_rho_boyd(self, iter):
        scale = (self.r_prim * self.eps_rel_dual) / (self.r_dual * self.eps_rel_prim)
        scale = np.sqrt(scale)
        self.scale_boyd = scale
        self.rho_estimate_boyd = scale * self.rho_boyd
        self.rho_estimate_boyd = min(
            max(self.rho_estimate_boyd, self.rho_min), self.rho_max
        )

        if (iter) % self.rho_update_interval == 0 and iter > 1:
            if (
                self.rho_estimate_boyd > self.rho_boyd * self.adaptive_rho_tolerance
                or self.rho_estimate_boyd < self.rho_boyd / self.adaptive_rho_tolerance
            ):
                self.rho_boyd = self.rho_estimate_boyd
                for i in range(len(self.lboyd)):
                    if self.lboyd[i] == -np.inf and self.uboyd[i] == np.inf:
                        self.rho_vec_boyd[i] = self.rho_min
                    elif abs(self.lboyd[i] - self.uboyd[i]) < 1e-4:
                        self.rho_vec_boyd[i] = 1e3 * self.rho_boyd
                    elif self.lboyd[i] != self.uboyd[i]:
                        self.rho_vec_boyd[i] = self.rho_boyd
                    else:
                        assert False  # safety check

    def set_rho_boyd(self):
        self.rho_vec_boyd = np.zeros(self.n_in)
        self.rho_boyd = min(max(self.rho_boyd, self.rho_min), self.rho_max)
        for i in range(len(self.lboyd)):
            if self.lboyd[i] == -np.inf and self.uboyd[i] == np.inf:
                self.rho_vec_boyd[i] = self.rho_min
            elif self.lboyd[i] == self.uboyd[i]:
                self.rho_vec_boyd[i] = 1e3 * self.rho_boyd
            elif self.lboyd[i] != self.uboyd[i]:
                self.rho_vec_boyd[i] = self.rho_boyd
        self.rho_estimate_boyd = 0

    def optimize_boyd(self, maxiters=1000):
        self.set_rho_boyd()
        converged = False
        for iter in range(1, maxiters + 1):
            self.solve_linear_system_boyd()
            self.update_lagrangian_params_boyd()
            self.update_rho_boyd(iter)

            eps_prim = self.eps_abs + self.eps_rel * self.eps_rel_prim
            eps_dual = self.eps_abs + self.eps_rel * self.eps_rel_dual

            if (iter) % self.rho_update_interval == 0 and iter > 1:
                if self.r_prim <= eps_prim and self.r_dual <= eps_dual:
                    if self.verboseQP:
                        print(
                            "Iters",
                            iter,
                            "res-primal",
                            pp(self.r_prim),
                            "res-dual",
                            pp(self.r_dual),
                            "optimal rho estimate",
                            pp(self.rho_estimate_boyd),
                            "rho",
                            pp(self.rho_boyd),
                        )
                        print("terminated ... \n")
                    converged = True
                    break
                if self.verboseQP:
                    print(
                        "Iters",
                        iter,
                        "res-primal",
                        pp(self.r_prim),
                        "res-dual",
                        pp(self.r_dual),
                        "optimal rho estimate",
                        pp(self.rho_estimate_boyd),
                        "rho",
                        pp(self.rho_boyd),
                    )
        self.qp_iters = iter
        if not converged and self.verboseQP:
            print("Not Converged ... \n")
            print(
                "Iters",
                iter,
                "res-primal",
                pp(self.r_prim),
                "res-dual",
                pp(self.r_dual),
                "optimal rho estimate",
                pp(self.rho_estimate_boyd),
                "rho",
                pp(self.rho_boyd),
            )
        return self.x_k.copy()
