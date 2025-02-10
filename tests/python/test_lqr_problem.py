"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that all methods converge in one iteration on a LQR problem.
"""

import os
import pathlib

import mim_solvers
import numpy as np

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from problems import create_lqr_problem  # noqa: E402
from sqp import SQP  # noqa: E402
from sqp_cpp import SQP_CPP  # noqa: E402

LINE_WIDTH = 100


print("TEST 1: python SQP = mim_solvers SQP".center(LINE_WIDTH, "-"))

problem, xs_init, us_init = create_lqr_problem()
ddp0 = SQP(problem)
ddp1 = SQP_CPP(problem)
ddp2 = mim_solvers.SolverSQP(problem)

ddp0.with_callbacks = True
ddp1.with_callbacks = True
ddp2.with_callbacks = True

ddp0.termination_tolerance = 1e-6
ddp1.termination_tolerance = 1e-6
ddp2.termination_tolerance = 1e-6


converged = ddp0.solve(xs_init, us_init, 10)
converged = ddp1.solve(xs_init, us_init, 10)
converged = ddp2.solve(xs_init, us_init, 10)

tol = 1e-4
assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp2.xs)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp2.us)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < tol, "Test failed"


assert ddp0.iter == 0
assert ddp1.iter == 0
assert ddp2.iter == 1


print("TEST PASSED".center(LINE_WIDTH, "-"))
print("\n")
