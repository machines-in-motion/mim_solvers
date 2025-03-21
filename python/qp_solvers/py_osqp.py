## Custom osqp implementation
## Author : Avadesh Meduri
## Date : 31/03/2023

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

np.set_printoptions(precision=2)
# np.format_float_scientific(self.r_prim, exp_digits=2, precision =2)


def pp(s):
    return np.format_float_scientific(s, exp_digits=2, precision=2)


class CustomOSQP:
    def __init__(self):
        self.sigma_osqp = 1e-6
        self.rho_osqp_init = 1e-1
        self.rho_min = 1e-6
        self.rho_max = 1e6
        self.alpha_osqp = 1.6

        self.eps_abs = 1e-3
        self.eps_rel = 1e-3
        self.adaptive_rho_tolerance = 5
        self.rho_update_interval = 25

    def solve_linear_system_osqp(self):
        ## More efficient
        rho_mat = self.rho_vec_osqp * np.eye(len(self.rho_vec_osqp))
        A_block = (
            self.P
            + self.sigma_osqp * np.eye(self.n_vars)
            + (self.Aosqp.T @ rho_mat @ self.Aosqp)
        )
        A_block = sparse.csr_matrix(A_block)
        b_block = (
            self.sigma_osqp * self.x_k
            - self.q
            + self.Aosqp.T @ (np.multiply(self.rho_vec_osqp, self.z_k) - self.y_k)
        )

        self.xtilde_k_1 = spsolve(A_block, b_block)
        self.ztilde_k_1 = self.Aosqp @ self.xtilde_k_1

    def update_lagrangian_params(self):
        self.x_k_1 = (
            self.alpha_osqp * self.xtilde_k_1 + (1 - self.alpha_osqp) * self.x_k
        )
        self.z_k_1 = np.clip(
            self.alpha_osqp * self.ztilde_k_1
            + (1 - self.alpha_osqp) * self.z_k
            + np.divide(self.y_k, self.rho_vec_osqp),
            self.losqp,
            self.uosqp,
        )
        self.y_k_1 = self.y_k + np.multiply(
            self.rho_vec_osqp,
            (
                self.alpha_osqp * self.ztilde_k_1
                + (1 - self.alpha_osqp) * self.z_k
                - self.z_k_1
            ),
        )

        # dual_vec = np.multiply(self.rho_vec_osqp, (self.z_k_1 - self.z_k))
        # self.r_dual = max(abs(dual_vec))
        self.x_k, self.z_k, self.y_k = self.x_k_1, self.z_k_1, self.y_k_1
        # self.r_prim = max(abs(self.ztilde_k_1 - self.z_k))

        self.r_prim = max(abs(self.Aosqp @ self.x_k - self.z_k))
        self.r_dual = max(abs(self.P @ self.x_k + self.q + self.Aosqp.T @ self.y_k))

        self.eps_rel_prim = max(abs(np.hstack((self.Aosqp @ self.x_k, self.z_k))))
        tmp = max(abs(self.q))
        tmp2 = max(abs(self.Aosqp.T @ self.y_k))
        tmp3 = max(abs(self.P @ self.x_k))
        self.eps_rel_dual = max(tmp, tmp2)
        self.eps_rel_dual = max(self.eps_rel_dual, tmp3)

    def update_rho_osqp(self, iter):
        scale = np.sqrt(
            self.r_prim * self.eps_rel_dual / (self.r_dual * self.eps_rel_prim)
        )
        self.rho_estimate_osqp = scale * self.rho_osqp
        self.rho_estimate_osqp = min(
            max(self.rho_estimate_osqp, self.rho_min), self.rho_max
        )

        if (iter) % self.rho_update_interval == 0 and iter > 1:
            if (
                self.rho_estimate_osqp > self.rho_osqp * self.adaptive_rho_tolerance
                or self.rho_estimate_osqp < self.rho_osqp / self.adaptive_rho_tolerance
            ):
                self.rho_osqp = self.rho_estimate_osqp
                for i in range(len(self.losqp)):
                    if self.losqp[i] == -np.inf and self.uosqp[i] == np.inf:
                        self.rho_vec_osqp[i] = self.rho_min
                    elif abs(self.losqp[i] - self.uosqp[i]) < 1e-4:
                        self.rho_vec_osqp[i] = 1e3 * self.rho_osqp
                    elif self.losqp[i] != self.uosqp[i]:
                        self.rho_vec_osqp[i] = self.rho_osqp
                    else:
                        assert False  # safety check

    def set_rho_osqp(self):
        self.rho_vec_osqp = np.zeros(self.n_eq + self.n_in)
        self.rho_osqp = min(max(self.rho_osqp_init, self.rho_min), self.rho_max)
        for i in range(len(self.losqp)):
            if self.losqp[i] == -np.inf and self.uosqp[i] == np.inf:
                self.rho_vec_osqp[i] = self.rho_min
            elif self.losqp[i] == self.uosqp[i]:
                self.rho_vec_osqp[i] = 1e3 * self.rho_osqp
            elif self.losqp[i] != self.uosqp[i]:
                self.rho_vec_osqp[i] = self.rho_osqp
        self.rho_estimate_osqp = 0

    def optimize_osqp(self, maxiters=1000):
        self.set_rho_osqp()
        self.converged = False
        for iter in range(1, maxiters + 1):
            self.solve_linear_system_osqp()
            self.update_lagrangian_params()
            self.update_rho_osqp(iter)

            eps_prim = self.eps_abs + self.eps_rel * self.eps_rel_prim
            eps_dual = self.eps_abs + self.eps_rel * self.eps_rel_dual

            if (iter) % self.rho_update_interval == 0 and iter > 1:
                if self.verboseQP:
                    print(
                        "Iters",
                        iter,
                        "res-primal",
                        pp(self.r_prim),
                        "res-dual",
                        pp(self.r_dual),
                        "optimal rho estimate",
                        pp(self.rho_estimate_osqp),
                        "rho",
                        pp(self.rho_osqp),
                    )
                if self.r_prim < eps_prim and self.r_dual < eps_dual:
                    if self.verboseQP:
                        print(
                            "Iters",
                            iter,
                            "res-primal",
                            pp(self.r_prim),
                            "res-dual",
                            pp(self.r_dual),
                            "optimal rho estimate",
                            pp(self.rho_estimate_osqp),
                            "rho",
                            pp(self.rho_osqp),
                        )
                        print("terminated ... \n")
                    self.converged = True
                    break
        self.qp_iters = iter
        if not self.converged and self.verboseQP:
            print(
                "Iters",
                iter,
                "res-primal",
                pp(self.r_prim),
                "res-dual",
                pp(self.r_dual),
                "optimal rho estimate",
                pp(self.rho_estimate_osqp),
                "rho",
                pp(self.rho_osqp),
            )
        return self.x_k
