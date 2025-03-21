"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that the python dense ADMM implementation matches the stagewise implementations of python and c++.
"""

import os
import pathlib

import mim_solvers
import numpy as np

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from csqp import CSQP  # noqa: E402
from problems import create_clqr_problem  # noqa: E402

LINE_WIDTH = 100

print(" TEST Stagewise ADMM PROBLEM ".center(LINE_WIDTH, "-"))

problem, xs_init, us_init = create_clqr_problem()

ddp1 = mim_solvers.SolverCSQP(problem)
ddp1.remove_reg = True
ddp1.reset_rho = True  # To-do: make sure that the python has the same default arguments
ddp1.reset_y = True


ddp2 = CSQP(problem, "StagewiseQP")
ddp3 = CSQP(problem, "StagewiseQPKKT")

ddp1.with_callbacks = True
ddp2.with_callbacks = True
ddp3.with_callbacks = True

termination_tolerance = 1e-4
ddp1.termination_tolerance = termination_tolerance
ddp2.termination_tolerance = termination_tolerance
ddp3.termination_tolerance = termination_tolerance

max_qp_iters = 1000
ddp1.max_qp_iters = max_qp_iters
ddp2.max_qp_iters = max_qp_iters
ddp3.max_qp_iters = max_qp_iters

eps_abs = 1e-8
eps_rel = 0.0
ddp1.eps_abs = eps_abs
ddp2.eps_abs = eps_abs
ddp3.eps_abs = eps_abs
ddp1.eps_rel = eps_rel
ddp2.eps_rel = eps_rel
ddp3.eps_rel = eps_rel


converged = ddp1.solve(xs_init, us_init, 1)
converged = ddp2.solve(xs_init, us_init, 1)
converged = ddp3.solve(xs_init, us_init, 1)


assert ddp1.qp_iters == ddp2.qp_iters
set_tol = 1e-6
assert np.linalg.norm(np.array(ddp1.dx_tilde) - np.array(ddp2.dx_tilde)) < set_tol, (
    "Test failed"
)
assert np.linalg.norm(np.array(ddp1.du_tilde) - np.array(ddp2.du_tilde)) < set_tol, (
    "Test failed"
)


assert np.linalg.norm(ddp1.norm_primal - np.array(ddp2.norm_primal)) < set_tol, (
    "Test failed"
)
assert np.linalg.norm(ddp1.norm_dual - np.array(ddp2.norm_dual)) < set_tol, (
    "Test failed"
)
assert (
    np.linalg.norm(ddp1.norm_primal_rel - np.array(ddp2.norm_primal_rel)) < set_tol
), "Test failed"
assert np.linalg.norm(ddp1.norm_dual_rel - np.array(ddp2.norm_dual_rel)) < set_tol, (
    "Test failed"
)

assert np.linalg.norm(ddp1.rho_sparse - np.array(ddp2.rho_sparse)) < set_tol, (
    "Test failed"
)


for t in range(len(ddp1.rho_vec)):
    assert np.linalg.norm(ddp1.rho_vec[t] - ddp2.rho_vec[t]) < set_tol, "Test failed"
    assert np.linalg.norm(ddp1.y[t] - ddp2.y[t]) < set_tol, "Test failed"
    assert np.linalg.norm(ddp1.z[t] - ddp2.z[t]) < set_tol, "Test failed"
