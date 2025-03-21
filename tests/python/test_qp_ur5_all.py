"""__init__

License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that the stagewise QP solver matches the other QP solvers implemented in python
on the ur5 example (with constraints)
"""

import importlib.util
import os
import pathlib
import time

import crocoddyl
import example_robot_data
import mim_solvers
import numpy as np

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from csqp import CSQP  # noqa: E402

HPIPM_PYTHON_FOUND = importlib.util.find_spec("hpipm_python")

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
N_ocp = T
xs_init = [x0] * (N_ocp + 1)
us_init = problem.quasiStatic(
    [problem.x0] * problem.T
)  # [np.zeros( actuation.nu)] * N_ocp


# Define solver
ddp0 = mim_solvers.SolverCSQP(problem)  # CSQP(problem, "StagewiseQP")
ddp1 = CSQP(problem, "ProxQP")
ddp2 = CSQP(problem, "OSQP")
if HPIPM_PYTHON_FOUND is not None:
    ddp4 = CSQP(problem, "HPIPM_DENSE")
    ddp5 = CSQP(problem, "HPIPM_OCP")

ddp0.with_callbacks = False
ddp1.with_callbacks = False
ddp2.with_callbacks = False
if HPIPM_PYTHON_FOUND is not None:
    ddp4.with_callbacks = False
    ddp5.with_callbacks = False

max_qp_iters = 10000
ddp0.max_qp_iters = max_qp_iters
ddp1.max_qp_iters = max_qp_iters
ddp2.max_qp_iters = max_qp_iters
if HPIPM_PYTHON_FOUND is not None:
    ddp4.max_qp_iters = max_qp_iters
    ddp5.max_qp_iters = max_qp_iters

eps_abs = 1e-10
eps_rel = 0.0
ddp0.eps_abs = eps_abs
ddp0.eps_rel = eps_rel
ddp1.eps_abs = eps_abs
ddp1.eps_rel = eps_rel
ddp2.eps_abs = eps_abs
ddp2.eps_rel = eps_rel
if HPIPM_PYTHON_FOUND is not None:
    ddp4.eps_abs = eps_abs
    ddp4.eps_rel = eps_rel
    ddp5.eps_abs = eps_abs
    ddp5.eps_rel = eps_rel


ddp0.equality_qp_initial_guess = False
ddp1.equality_qp_initial_guess = False
ddp2.equality_qp_initial_guess = False
if HPIPM_PYTHON_FOUND is not None:
    ddp4.equality_qp_initial_guess = False
    ddp5.equality_qp_initial_guess = False

ddp0.update_rho_with_heuristic = True


# Stagewise QP
print("\n ------ STAGEWISE (computeDirection)------ ")
converged = ddp0.solve(xs_init, us_init, 0)
t0 = time.time()
ddp0.computeDirection(True)
print("Stagewise : ", time.time() - t0)
print("------------------------ \n")

# ProxQP
print("\n ------ ProxQP ------ ")
converged = ddp1.solve(xs_init, us_init, 1)
print("------------------------ \n")


# OSQP
print("\n ------ OSQP ------ ")
converged = ddp2.solve(xs_init, us_init, 1)
print("------------------------ \n")

if HPIPM_PYTHON_FOUND is not None:
    # HPIPM
    print("\n ------ HPIPM DENSE ------ ")
    converged = ddp4.solve(xs_init, us_init, 1)
    print("------------------------ \n")

    # HPIPM
    print("\n ------ HPIPM OCP ------ ")
    converged = ddp5.solve(xs_init, us_init, 1)
    print("------------------------ \n")

#  iterations
print("Stagewise iter = ", int(ddp0.qp_iters))
print("ProxQP iter = ", int(ddp1.qp_iters))
print("OSQP iter      = ", ddp2.qp_iters)
if HPIPM_PYTHON_FOUND is not None:
    print("HPIPM DENSE iter     = ", ddp4.qp_iters)
    print("HPIPM OCP iter     = ", ddp5.qp_iters)

# Check that QP solutions are the same
TOL = 1e-2
assert np.linalg.norm(np.array(ddp0.dx_tilde.tolist()) - np.array(ddp1.dx)) < TOL
assert np.linalg.norm(np.array(ddp0.dx_tilde.tolist()) - np.array(ddp2.dx)) < TOL
if HPIPM_PYTHON_FOUND is not None:
    assert np.linalg.norm(np.array(ddp0.dx_tilde.tolist()) - np.array(ddp4.dx)) < TOL
    assert np.linalg.norm(np.array(ddp0.dx_tilde.tolist()) - np.array(ddp5.dx)) < TOL
    assert np.linalg.norm(np.array(ddp4.dx) - np.array(ddp5.dx)) < TOL
assert np.linalg.norm(np.array(ddp0.du_tilde.tolist()) - np.array(ddp1.du)) < TOL
assert np.linalg.norm(np.array(ddp0.du_tilde.tolist()) - np.array(ddp2.du)) < TOL
if HPIPM_PYTHON_FOUND is not None:
    assert np.linalg.norm(np.array(ddp0.du_tilde.tolist()) - np.array(ddp4.du)) < TOL
    assert np.linalg.norm(np.array(ddp0.du_tilde.tolist()) - np.array(ddp5.du)) < TOL
    assert np.linalg.norm(np.array(ddp4.du) - np.array(ddp5.du)) < TOL
