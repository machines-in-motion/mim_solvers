import pathlib
import os
import sys
import numpy as np
import crocoddyl
import pinocchio as pin

python_path = pathlib.Path('.').absolute().parent/'python'
print(python_path)
os.sys.path.insert(1, str(python_path))
from py_mim_solvers import SQP

from problems import create_double_pendulum_problem, create_quadrotor_problem
import mim_solvers

# Solver params
MAXITER     = 3
TOL         = 1e-4 
CALLBACKS   = True
FILTER_SIZE = MAXITER

# Create 1 solver of each type for each problem


# pb = create_double_pendulum_problem()
pb = create_quadrotor_problem()

x0 = pb.x0.copy()

# # Create solver SQP (MS)
solverSQP = mim_solvers.SolverSQP(pb)
solverSQP.xs = [solverSQP.problem.x0] * (solverSQP.problem.T + 1)  
solverSQP.us = solverSQP.problem.quasiStatic([solverSQP.problem.x0] * solverSQP.problem.T)
solverSQP.termination_tol        = TOL
solverSQP.use_filter_line_search = True
solverSQP.filter_size            = MAXITER
solverSQP.with_callbacks         = CALLBACKS
solverSQP.xReg = 0.0

# # SQP        
solverSQP.xs = [x0] * (pb.T + 1)
solverSQP.us = pb.quasiStatic([x0] * pb.T)
# solverSQP.computeDirection(True)
solverSQP.solve([x0] * (pb.T + 1) , pb.quasiStatic([x0] * pb.T), MAXITER, False)

# # Check convergence
# solved = (solverSQP.iter < MAXITER) and (solverSQP.KKT < TOL)
# print(solved)

# python solver
pysolverSQP = SQP(pb, VERBOSE = True)
pysolverSQP.termination_tol             = TOL


pysolverSQP.xs = [x0] * (pb.T + 1) 
pysolverSQP.us = pb.quasiStatic([x0] * pb.T)
pysolverSQP.computeDirection()
# print(pysolverSQP.cost)
pysolverSQP.solve(pysolverSQP.xs.copy(), pysolverSQP.us.copy(), MAXITER)

##### UNIT TEST #####################################

set_tol = 1e-4

for t in range(pb.T-1, 0, -1):
    assert np.linalg.norm(pysolverSQP.L[t] + solverSQP.K[t]) < set_tol
    assert np.linalg.norm(pysolverSQP.l[t] + solverSQP.k[t]) < set_tol

assert np.linalg.norm(np.array(pysolverSQP.xs) - np.array(solverSQP.xs))/pb.T < set_tol, "Test failed"
assert np.linalg.norm(np.array(pysolverSQP.us) - np.array(solverSQP.us))/pb.T < set_tol, "Test failed"

assert pysolverSQP.KKT - solverSQP.KKT < set_tol, "Test failed"

print("ALL UNIT TEST PASSED .....")