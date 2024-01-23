"""__init__

License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
import pathlib
import os
python_path = pathlib.Path('.').absolute().parent
os.sys.path.insert(1, str(python_path))
import crocoddyl
import numpy as np
import mim_solvers
import example_robot_data
from sqp_cpp import SQP_CPP
from sqp import SQP

LINE_WIDTH = 100


# # # # # # # # # # # # # # #
###       LOAD ROBOT      ###
# # # # # # # # # # # # # # #

robot = example_robot_data.load("ur5")
model = robot.model
nq = model.nq
nv = model.nv
nu = nv
q0 = np.array([0, 0, 0, 0, 0, 0])
v0 = np.zeros(nv)
x0 = np.concatenate([q0, v0]).copy()

# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

# Create cost terms
# Control regularization cost
uResidual = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
# State regularization cost
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
# endeff frame translation cost
endeff_frame_id = model.getFrameId("tool0")
endeff_translation = np.array([0.4, 0.4, 0.4])
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
    state, endeff_frame_id, endeff_translation
)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)

# Create the running models
runningModels = []
dt = 5e-2
T = 40
for t in range(T + 1):
    runningCostModel = crocoddyl.CostModelSum(state)
    # Add costs
    runningCostModel.addCost("stateReg", xRegCost, 1e-1)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    if t != T:
        runningCostModel.addCost("translation", frameTranslationCost, 4)
    else:
        runningCostModel.addCost("translation", frameTranslationCost, 40)

    # Create Differential action model
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, runningCostModel
    )
    # Apply Euler integration
    running_model = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
    runningModels.append(running_model)


# Create the shooting problem
problem = crocoddyl.ShootingProblem(x0, runningModels[:-1], runningModels[-1])


# # # # # # # # # # # # #
###     SOLVE OCP     ###
# # # # # # # # # # # # #

# Define warm start
xs = [x0] * (T + 1)
us = [np.zeros(nu)] * T


# Create solvers
ddp0 = SQP(problem)
ddp1 = SQP_CPP(problem)
ddp2 = mim_solvers.SolverSQP(problem)

# Set Filter Line Search
ddp0.use_filter_line_search = True
ddp1.use_filter_line_search = True
ddp2.use_filter_line_search = True

# Set filter size
ddp0.filter_size = 10
ddp1.filter_size = 10
ddp2.filter_size = 10

# Set callbacks
ddp0.with_callbacks = True
ddp1.with_callbacks = True
ddp2.with_callbacks = True

# Set tolerance 
ddp0.termination_tolerance = 1e-8
ddp1.termination_tolerance = 1e-8
ddp2.termination_tolerance = 1e-8


# Solve
ddp0.solve(xs, us, 100, False)
ddp1.solve(xs, us, 100, False)
ddp2.solve(xs, us, 100, False)

# ##### UNIT TEST #####################################

tol = 1e-4
assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp1.xs)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp1.us)) < tol, "Test failed"

assert np.linalg.norm(np.array(ddp0.xs) - np.array(ddp2.xs)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.us) - np.array(ddp2.us)) < tol, "Test failed"

assert ddp0.cost - ddp1.cost < tol, "Test failed"
assert ddp0.cost - ddp2.cost < tol, "Test failed"

assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp1.lag_mul)) < tol, "Test failed"
assert np.linalg.norm(np.array(ddp0.lag_mul) - np.array(ddp2.lag_mul)) < tol, "Test failed"

assert ddp0.KKT - ddp1.KKT < tol, "Test failed"
assert ddp0.KKT - ddp2.KKT < tol, "Test failed"

print("TEST PASSED".center(LINE_WIDTH, "-"))
print("\n")