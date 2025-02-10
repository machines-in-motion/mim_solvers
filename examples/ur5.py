"""__init__

License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import crocoddyl
import example_robot_data
import matplotlib.pyplot as plt
import mim_solvers
import numpy as np

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
xs = [x0] * (T + 1)
us = [np.zeros(nu)] * T

# Define solver
solver = mim_solvers.SolverCSQP(problem)
solver.termination_tolerance = 1e-4
# solver.with_callbacks = True
solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])

# Solve
max_iter = 100
solver.solve(xs, us, max_iter)
log = solver.getCallbacks()[-1]
crocoddyl.plotOCSolution(solver.xs, solver.us)
mim_solvers.plotConvergence(log.convergence_data)


x_traj = np.array(solver.xs)
u_traj = np.array(solver.us)
p_traj = np.zeros((len(solver.xs), 3))

for i in range(T + 1):
    robot.framesForwardKinematics(x_traj[i, :nq])
    p_traj[i] = robot.data.oMf[endeff_frame_id].translation


time_lin = np.linspace(0, dt * (T + 1), T + 1)

fig, axs = plt.subplots(nq)
for i in range(nq):
    axs[i].plot(time_lin, x_traj[:, i])
fig.suptitle("State trajectory")


fig, axs = plt.subplots(nq)
for i in range(nq):
    axs[i].plot(time_lin[:-1], u_traj[:, i])
fig.suptitle("Control trajectory")


fig, axs = plt.subplots(3)
for i in range(3):
    axs[i].plot(time_lin, p_traj[:, i])
    axs[i].plot(time_lin[-1], endeff_translation[i], "o")
fig.suptitle("End effector trajectory")
plt.show()

# viewer
WITHDISPLAY = True
if WITHDISPLAY:
    import time

    display = crocoddyl.MeshcatDisplay(robot)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
