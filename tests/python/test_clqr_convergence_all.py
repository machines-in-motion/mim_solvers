"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that all methods converge in one iteration on a constrained LQR problem.
"""

import importlib.util
import os
import pathlib

import mim_solvers
import numpy as np

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from csqp import CSQP  # noqa: E402
from problems import create_clqr_problem  # noqa: E402

HPIPM_PYTHON_FOUND = importlib.util.find_spec("hpipm_python")

LINE_WIDTH = 100

problem, xs_init, us_init = create_clqr_problem()

print(" TEST CLQR PROBLEM ".center(LINE_WIDTH, "-"))

ddp1 = mim_solvers.SolverCSQP(problem)
ddp2 = CSQP(problem, "StagewiseQP")
ddp3 = CSQP(problem, "CustomOSQP")
ddp4 = CSQP(problem, "StagewiseQPKKT")
ddp5 = CSQP(problem, "OSQP")
ddp6 = CSQP(problem, "ProxQP")
if HPIPM_PYTHON_FOUND is not None:
    ddp7 = CSQP(problem, "HPIPM_DENSE")
    ddp8 = CSQP(problem, "HPIPM_OCP")

ddp1.with_callbacks = False
ddp2.with_callbacks = False
ddp3.with_callbacks = False
ddp4.with_callbacks = False
ddp5.with_callbacks = False
ddp6.with_callbacks = False
if HPIPM_PYTHON_FOUND is not None:
    ddp7.with_callbacks = False
    ddp8.with_callbacks = False

termination_tolerance = 1e-4
ddp1.termination_tolerance = termination_tolerance
ddp2.termination_tolerance = termination_tolerance
ddp3.termination_tolerance = termination_tolerance
ddp4.termination_tolerance = termination_tolerance
ddp5.termination_tolerance = termination_tolerance
ddp6.termination_tolerance = termination_tolerance
if HPIPM_PYTHON_FOUND is not None:
    ddp7.termination_tolerance = termination_tolerance
    ddp8.termination_tolerance = termination_tolerance

max_qp_iters = 10000
ddp1.max_qp_iters = max_qp_iters
ddp2.max_qp_iters = max_qp_iters
ddp3.max_qp_iters = max_qp_iters
ddp4.max_qp_iters = max_qp_iters
ddp5.max_qp_iters = max_qp_iters
ddp6.max_qp_iters = max_qp_iters
if HPIPM_PYTHON_FOUND is not None:
    ddp7.max_qp_iters = max_qp_iters
    ddp8.max_qp_iters = max_qp_iters

eps_abs = 1e-8
eps_rel = 0.0
ddp1.eps_abs = eps_abs
ddp2.eps_abs = eps_abs
ddp3.eps_abs = eps_abs
ddp4.eps_abs = eps_abs
ddp5.eps_abs = eps_abs
ddp6.eps_abs = eps_abs
if HPIPM_PYTHON_FOUND is not None:
    ddp7.eps_abs = eps_abs
    ddp8.eps_abs = eps_abs
ddp1.eps_rel = eps_rel
ddp2.eps_rel = eps_rel
ddp3.eps_rel = eps_rel
ddp4.eps_rel = eps_rel
ddp5.eps_rel = eps_rel
ddp6.eps_rel = eps_rel
if HPIPM_PYTHON_FOUND is not None:
    ddp7.eps_rel = eps_rel
    ddp8.eps_rel = eps_rel

converged = ddp1.solve(xs_init, us_init, 2)
converged = ddp2.solve(xs_init, us_init, 2)
converged = ddp3.solve(xs_init, us_init, 2)
converged = ddp4.solve(xs_init, us_init, 2)
converged = ddp5.solve(xs_init, us_init, 2)
converged = ddp6.solve(xs_init, us_init, 2)
if HPIPM_PYTHON_FOUND is not None:
    converged = ddp7.solve(xs_init, us_init, 2)
    converged = ddp8.solve(xs_init, us_init, 2)

################################## TEST CONVERGENCE #####################################
set_tol = 1e-4
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.lag_mul) - np.array(ddp2.lag_mul)) < set_tol, (
    "Test failed"
)


assert (
    ddp1.iter == 1
)  # To-do: make sure python and c++ have the same logic in terms of iteration count
assert ddp2.iter == 0
assert ddp3.iter == 0
assert ddp4.iter == 0
assert ddp5.iter == 0
assert ddp6.iter == 0
if HPIPM_PYTHON_FOUND is not None:
    assert ddp7.iter == 0
    assert ddp8.iter == 0


assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp3.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp3.us)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp4.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp4.us)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp5.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp5.us)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp6.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp6.us)) < set_tol, "Test failed"

if HPIPM_PYTHON_FOUND is not None:
    assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp7.xs)) < set_tol, (
        "Test failed"
    )
    assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp7.us)) < set_tol, (
        "Test failed"
    )

    assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp8.xs)) < set_tol, (
        "Test failed"
    )
    assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp8.us)) < set_tol, (
        "Test failed"
    )
