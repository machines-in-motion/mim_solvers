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
from csqp import CSQP

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

# Create contraint on end-effector
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(
    state, endeff_frame_id, np.zeros(3)
)
ee_contraint = crocoddyl.ConstraintModelResidual(
    state,
    frameTranslationResidual,
    np.array([-1.0, -1.0, -1.0]),
    np.array([1., 0.4, 0.4]),
)

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
    # Define contraints
    constraints = crocoddyl.ConstraintModelManager(state, nu)
    if t != 0:
        constraints.addConstraint("ee_bound", ee_contraint)
    # Create Differential action model
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, runningCostModel, constraints
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
xs_init_1 = [x0] * (T + 1)
us_init_1 = [np.zeros(nu)] * T

xs_init_2 = [x0] * (T + 1)
us_init_2 = [np.zeros(nu)] * T

ddp1 = mim_solvers.SolverCSQP(problem)
ddp2 = CSQP(problem, "StagewiseQP")

ddp1.with_callbacks = True
ddp2.with_callbacks = True

termination_tolerance = 1e-3
ddp1.termination_tolerance = termination_tolerance
ddp2.termination_tolerance = termination_tolerance

ddp1.max_qp_iters = 2000
ddp2.max_qp_iters = 2000

ddp1.eps_abs = 1e-6
ddp1.eps_rel = 0.

ddp2.eps_abs = 1e-6
ddp2.eps_rel = 0.

converged = ddp1.solve(xs_init_1, us_init_1, 20)
converged = ddp2.solve(xs_init_2, us_init_2, 20)


##### UNIT TEST #####################################
set_tol = 1e-3
assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.dx_tilde) - np.array(ddp2.dx_tilde)) < set_tol, "Test failed"
assert np.linalg.norm(np.array(ddp1.du_tilde) - np.array(ddp2.du_tilde)) < set_tol, "Test failed"

assert np.linalg.norm(np.array(ddp1.lag_mul) - np.array(ddp2.lag_mul)) < set_tol, "Test failed"


assert ddp1.KKT < termination_tolerance, "Test failed"
assert ddp2.KKT < termination_tolerance, "Test failed"