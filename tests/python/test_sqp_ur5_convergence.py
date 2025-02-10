"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that all methods converge to the same solution on the ur5 example.
"""

import os
import pathlib

import mim_solvers
import numpy as np

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from problems import create_unconstrained_ur5  # noqa: E402
from sqp import SQP  # noqa: E402
from sqp_cpp import SQP_CPP  # noqa: E402

LINE_WIDTH = 100


# # # # # # # # # # # # # # #
###       LOAD ROBOT      ###
# # # # # # # # # # # # # # #

problem, xs, us = create_unconstrained_ur5()


# Create solvers
ddp0 = SQP(problem)
ddp1 = SQP_CPP(problem)
ddp2 = mim_solvers.SolverSQP(problem)

# Set Filter Line Search
ddp0.use_filter_line_search = True
ddp1.use_filter_line_search = True
ddp2.use_filter_line_search = True

# Set filter size
ddp0.filter_size = 10
ddp1.filter_size = 10
ddp2.filter_size = 10

# Set callbacks
ddp0.with_callbacks = True
ddp1.with_callbacks = True
ddp2.with_callbacks = True

# Set tolerance
ddp0.termination_tolerance = 1e-10
ddp1.termination_tolerance = 1e-10
ddp2.termination_tolerance = 1e-10

# Solve
ddp0.solve(xs, us, 100, False)
ddp1.solve(xs, us, 100, False)
ddp2.solve(xs, us, 100, False)

# ##### UNIT TEST #####################################

tol = 1e-6
assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp1.xs)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp1.us)) < tol, "Test failed"

assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp2.xs)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp2.us)) < tol, "Test failed"

assert ddp0.cost - ddp1.cost < tol, "Test failed"
assert ddp0.cost - ddp2.cost < tol, "Test failed"

assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp1.lag_mul)) < tol, (
    "Test failed"
)
assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp2.lag_mul)) < tol, (
    "Test failed"
)

assert ddp0.KKT - ddp1.KKT < tol, "Test failed"
assert ddp0.KKT - ddp2.KKT < tol, "Test failed"

print("TEST PASSED".center(LINE_WIDTH, "-"))
print("\n")
