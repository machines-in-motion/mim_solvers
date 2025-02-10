"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that the python and c++ csqp implementation match on ur5.
"""

import os
import pathlib

import crocoddyl
import example_robot_data
import mim_solvers
import numpy as np

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from csqp import CSQP  # noqa: E402

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
    np.array([1.0, 0.4, 0.4]),
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

ddp1.use_filter_line_search = True
ddp2.use_filter_line_search = True

max_sqp_iter = 4

ddp1.filter_size = max_sqp_iter
ddp2.filter_size = max_sqp_iter

ddp1.extra_iteration_for_last_kkt = True
ddp2.extra_iteration_for_last_kkt = True

termination_tolerance = 1e-8
ddp1.termination_tolerance = termination_tolerance
ddp2.termination_tolerance = termination_tolerance

ddp1.max_qp_iters = 1000
ddp2.max_qp_iters = 1000

ddp1.eps_abs = 1e-4
ddp1.eps_rel = 0.0

ddp2.eps_abs = 1e-4
ddp2.eps_rel = 0.0

# Remove regularization in cpp solver
ddp1.remove_reg = True


set_tol = 1e-8

for reset_rho in [True, False]:
    for reset_y in [True, False]:
        ddp1.reset_rho = reset_rho
        ddp1.reset_y = reset_y

        ddp2.reset_rho = reset_rho
        ddp2.reset_y = reset_y

        converged = ddp1.solve(xs_init_1, us_init_1, max_sqp_iter)
        converged = ddp2.solve(xs_init_2, us_init_2, max_sqp_iter)

        ##### UNIT TEST #####################################
        assert np.linalg.norm(np.array(ddp1.xs) - np.array(ddp2.xs)) < set_tol, (
            "Test failed"
        )
        assert np.linalg.norm(np.array(ddp1.us) - np.array(ddp2.us)) < set_tol, (
            "Test failed"
        )

        assert (
            np.linalg.norm(np.array(ddp1.dx_tilde) - np.array(ddp2.dx_tilde)) < set_tol
        ), "Test failed"
        assert (
            np.linalg.norm(np.array(ddp1.du_tilde) - np.array(ddp2.du_tilde)) < set_tol
        ), "Test failed"

        assert (
            np.linalg.norm(np.array(ddp1.lag_mul) - np.array(ddp2.lag_mul)) < set_tol
        ), "Test failed"

        assert ddp1.qp_iters == ddp2.qp_iters
