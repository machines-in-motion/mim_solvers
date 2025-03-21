"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that the python and c++ sqp implementation match without regularization.
"""

import os
import pathlib

import mim_solvers
import numpy as np

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from problems import (  # noqa: E402
    create_double_pendulum_problem,
    create_quadrotor_problem,
    create_taichi,
    create_unconstrained_ur5,
)
from sqp import SQP  # noqa: E402

# Solver params
MAXITER = 100
TOL = 1e-4
with_callbacks = True
FILTER_SIZE = MAXITER

# Create 1 solver of each type for each problem

problems = [
    create_double_pendulum_problem(),
    create_quadrotor_problem(),
    create_unconstrained_ur5(),
    create_taichi(),
]

for problem, xs_init, us_init in problems:
    x0 = problem.x0.copy()
    # Create solver SQP (MS)
    solverSQP = mim_solvers.SolverSQP(problem)
    solverSQP.termination_tolerance = TOL
    solverSQP.use_filter_line_search = True
    solverSQP.filter_size = MAXITER
    solverSQP.with_callbacks = with_callbacks
    solverSQP.reg_min = 0.0  # This turns of regularization completely.
    reginit = 0.0

    # Create python solver
    pysolverSQP = SQP(problem, with_callbacks=with_callbacks)
    pysolverSQP.termination_tolerance = TOL
    pysolverSQP.use_filter_line_search = True
    pysolverSQP.filter_size = MAXITER

    # SQP
    solverSQP.solve(xs_init, us_init, MAXITER, False, reginit)

    pysolverSQP.solve(xs_init, us_init, MAXITER)

##### UNIT TEST #####################################

set_tol = 1e-4

for t in range(problem.T - 1, 0, -1):
    assert (
        np.linalg.norm(pysolverSQP.L[t] + solverSQP.K[t], 1)
        < (np.size(pysolverSQP.L[t])) * set_tol
    )
    assert (
        np.linalg.norm(pysolverSQP.l[t] + solverSQP.k[t], 1) / (len(pysolverSQP.l[t]))
        < set_tol
    )

assert (
    np.linalg.norm(np.array(pysolverSQP.xs) - np.array(solverSQP.xs)) / (problem.T + 1)
    < set_tol
), "Test failed"
assert (
    np.linalg.norm(np.array(pysolverSQP.us) - np.array(solverSQP.us)) / problem.T
    < set_tol
), "Test failed"

assert pysolverSQP.KKT - solverSQP.KKT < set_tol, "Test failed"

print("ALL UNIT TEST PASSED .....")
