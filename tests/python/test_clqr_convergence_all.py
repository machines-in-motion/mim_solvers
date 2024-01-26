"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that all methods converge in one iteration on a constrained LQR problem.
"""


import pathlib
import os
python_path = pathlib.Path('.').absolute().parent.parent/'python'
os.sys.path.insert(1, str(python_path))

import numpy as np
from csqp import CSQP

import mim_solvers
from problems import create_clqr_problem

LINE_WIDTH = 100

problem, xs_init, us_init = create_clqr_problem()

print(" TEST CLQR PROBLEM ".center(LINE_WIDTH, "-"))

ddp1 = mim_solvers.SolverCSQP(problem)
ddp2 = CSQP(problem, "StagewiseQP")
ddp3 = CSQP(problem, "CustomOSQP")
ddp4 = CSQP(problem, "StagewiseQPKKT")
ddp5 = CSQP(problem, "OSQP")
ddp6 = CSQP(problem, "ProxQP")

ddp1.with_callbacks = True
ddp2.with_callbacks = True
ddp3.with_callbacks = True
ddp4.with_callbacks = True
ddp5.with_callbacks = True
ddp6.with_callbacks = True

termination_tolerance = 1e-4
ddp1.termination_tolerance = termination_tolerance
ddp2.termination_tolerance = termination_tolerance
ddp3.termination_tolerance = termination_tolerance
ddp4.termination_tolerance = termination_tolerance
ddp5.termination_tolerance = termination_tolerance
ddp6.termination_tolerance = termination_tolerance

max_qp_iters = 10000
ddp1.max_qp_iters = max_qp_iters
ddp2.max_qp_iters = max_qp_iters
ddp3.max_qp_iters = max_qp_iters
ddp4.max_qp_iters = max_qp_iters
ddp5.max_qp_iters = max_qp_iters
ddp6.max_qp_iters = max_qp_iters

eps_abs = 1e-5
eps_rel = 0.
ddp1.eps_abs = eps_abs
ddp2.eps_abs = eps_abs
ddp3.eps_abs = eps_abs
ddp4.eps_abs = eps_abs
ddp5.eps_abs = eps_abs
ddp6.eps_abs = eps_abs
ddp1.eps_rel = eps_rel
ddp2.eps_rel = eps_rel
ddp3.eps_rel = eps_rel
ddp4.eps_rel = eps_rel
ddp5.eps_rel = eps_rel
ddp6.eps_rel = eps_rel



converged = ddp1.solve(xs_init, us_init, 2)
converged = ddp2.solve(xs_init, us_init, 2)
converged = ddp3.solve(xs_init, us_init, 2)
converged = ddp4.solve(xs_init, us_init, 2)
converged = ddp5.solve(xs_init, us_init, 2)
converged = ddp6.solve(xs_init, us_init, 2)

################################## TEST CONVERGENCE #####################################
set_tol = 1e-4
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.lag_mul) - np.array(ddp2.lag_mul)) < set_tol, "Test failed"



assert ddp1.iter == 1      # To-do: make sure python and c++ have the same logic in terms of iteration count 
assert ddp2.iter == 0
assert ddp3.iter == 0
assert ddp4.iter == 0
assert ddp5.iter == 0
assert ddp6.iter == 0


assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp3.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp3.us)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp4.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp4.us)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp5.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp5.us)) < set_tol, "Test failed"


