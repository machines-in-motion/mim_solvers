### Kuka reaching example with different constraint implementation

import crocoddyl
import numpy as np
import pinocchio as pin
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=4, linewidth=180)
import mim_solvers

from problems import create_kuka_problem

# import pathlib
# import os
# python_path = pathlib.Path('.').absolute().parent/'python'
# os.sys.path.insert(1, str(python_path))


# Solver params
MAXITER     = 20
TOL         = 1e-4 
CALLBACKS   = True
FILTER_SIZE = MAXITER


problem = create_kuka_problem()
x0 = problem.x0

# Create solver SQP (MS)
solverSQP = mim_solvers.SolverCSQP(problem)
solverSQP.xs = [solverSQP.problem.x0] * (solverSQP.problem.T + 1)  
solverSQP.us = solverSQP.problem.quasiStatic([solverSQP.problem.x0] * solverSQP.problem.T)
solverSQP.termination_tol        = TOL
solverSQP.use_filter_line_search = True
solverSQP.filter_size            = MAXITER
solverSQP.with_callbacks         = CALLBACKS
# solverSQP.reg_min                = 0.0 # This turns of regularization completely. 
# reginit                          = 0.0

# SQP    
solverSQP.xs = [x0] * (problem.T + 1)
solverSQP.us = problem.quasiStatic([x0] * problem.T)
solverSQP.solve(solverSQP.xs.copy() , solverSQP.us.copy(), MAXITER, False)

