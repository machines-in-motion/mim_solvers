"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that all the custom OSQP implementation matches the official one.
"""

import os
import pathlib

import numpy as np

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from csqp import CSQP  # noqa: E402
from problems import create_clqr_problem  # noqa: E402

LINE_WIDTH = 100

print(" TEST OSQP ".center(LINE_WIDTH, "-"))

problem, xs_init, us_init = create_clqr_problem()

ddp1 = CSQP(problem, "CustomOSQP")
ddp2 = CSQP(problem, "OSQP")

ddp1.with_callbacks = True
ddp2.with_callbacks = True

max_qp_iters = 10000
ddp1.max_qp_iters = max_qp_iters
ddp2.max_qp_iters = max_qp_iters

eps_abs = 1e-8
eps_rel = 0.0
ddp1.eps_abs = eps_abs
ddp2.eps_abs = eps_abs

converged = ddp1.solve(xs_init, us_init, 1)
converged = ddp2.solve(xs_init, us_init, 1)


assert ddp1.qp_iters == ddp2.qp_iters

set_tol = 1e-8
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.lag_mul) - np.array(ddp2.lag_mul)) < set_tol, (
    "Test failed"
)
for t in range(len(ddp1.y)):
    assert np.linalg.norm(ddp1.y[t] - ddp2.y[t]) < set_tol, "Test failed"
