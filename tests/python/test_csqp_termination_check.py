"""
License: BSD 3-Clause License

This file checks termination_check_interval: the ADMM residuals checked more
often stop the QP earlier, on the same solution.
"""

import mim_solvers
import numpy as np

from problems import create_clqr_problem  # noqa: E402


def solve(check_interval):
    problem, xs_init, us_init = create_clqr_problem()
    solver = mim_solvers.SolverCSQP(problem)
    solver.termination_tolerance = 1e-4
    solver.eps_abs = 1e-4  # default: 275 QP iterations, every one checked: 258
    solver.eps_rel = 0.0
    solver.max_qp_iters = 1000
    if check_interval is not None:
        solver.termination_check_interval = check_interval
    solver.solve(xs_init, us_init, 1, False)
    xs = np.array([np.array(x) for x in solver.xs])  # copies: temporaries
    us = np.array([np.array(u) for u in solver.us])
    return solver, xs, us


default, xs, us = solve(None)
assert default.termination_check_interval == 0  # every rho_update_interval
assert default.qp_iters % default.rho_update_interval == 0

checked, xs_checked, us_checked = solve(1)
assert checked.termination_check_interval == 1
assert checked.qp_iters < default.qp_iters
assert np.allclose(xs_checked, xs, atol=1e-3)
assert np.allclose(us_checked, us, atol=1e-3)

# Every 5 iterations: between the two.
five, _, _ = solve(5)
assert five.qp_iters % 5 == 0 and checked.qp_iters <= five.qp_iters < default.qp_iters
