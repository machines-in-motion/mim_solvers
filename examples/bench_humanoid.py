"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that all the custom OSQP implementation matches the official one.
"""
"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that all the custom OSQP implementation matches the official one.
"""


import pathlib
import os
python_path = pathlib.Path('.').absolute().parent/'python'
os.sys.path.insert(1, str(python_path))
import numpy as np
from csqp import CSQP
import mim_solvers
import example_robot_data
import crocoddyl



import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio
import crocoddyl
from crocoddyl.utils.biped import plotSolution

import mim_solvers

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
USE_FDDP = "use_fddp" in sys.argv   # Use SQP by default

signal.signal(signal.SIGINT, signal.SIG_DFL)

# Load robot
robot = example_robot_data.load("talos")
rmodel = robot.model
lims = rmodel.effortLimit
# lims[19:] *= 0.5  # reduced artificially the torque limits
rmodel.effortLimit = lims

# Create data structures
rdata = rmodel.createData()
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Set integration time
DT = 5e-2
T = 40
target = np.array([0.4, 0, 1.2])

# Initialize reference state, target and reference CoM
rightFoot = "right_sole_link"
leftFoot = "left_sole_link"
endEffector = "gripper_left_joint"
endEffectorId = rmodel.getFrameId(endEffector)
rightFootId = rmodel.getFrameId(rightFoot)
leftFootId = rmodel.getFrameId(leftFoot)
q0 = rmodel.referenceConfigurations["half_sitting"]
x0 = np.concatenate([q0, np.zeros(rmodel.nv)])
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
rfPos0 = rdata.oMf[rightFootId].translation
lfPos0 = rdata.oMf[leftFootId].translation
refGripper = rdata.oMf[rmodel.getFrameId("gripper_left_joint")].translation
comRef = (rfPos0 + lfPos0) / 2
comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()

# Initialize viewer
display = None
if WITHDISPLAY:
    if display is None:
        try:
            import gepetto

            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(robot, frameNames=[rightFoot, leftFoot])
            display.robot.viewer.gui.addSphere(
                "world/point", 0.05, [1.0, 0.0, 0.0, 1.0]
            )  # radius = .1, RGBA=1001
            display.robot.viewer.gui.applyConfiguration(
                "world/point", [*target.tolist(), 0.0, 0.0, 0.0, 1.0]
            )  # xyz+quaternion
        except Exception:
            display = crocoddyl.MeshcatDisplay(robot, frameNames=[rightFoot, leftFoot])

# Create two contact models used along the motion
contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
supportContactModelLeft = crocoddyl.ContactModel6D(
    state,
    leftFootId,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL,
    actuation.nu,
    np.array([0, 40]),
)
supportContactModelRight = crocoddyl.ContactModel6D(
    state,
    rightFootId,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL,
    actuation.nu,
    np.array([0, 40]),
)
contactModel1Foot.addContact(rightFoot + "_contact", supportContactModelRight)
contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)

# Cost for self-collision
maxfloat = sys.float_info.max
xlb = np.concatenate(
    [
        -maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.lowerPositionLimit[7:],
        -maxfloat * np.ones(state.nv),
    ]
)
xub = np.concatenate(
    [
        maxfloat * np.ones(6),  # dimension of the SE(3) manifold
        rmodel.upperPositionLimit[7:],
        maxfloat * np.ones(state.nv),
    ]
)
bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
xLimitResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xLimitActivation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
limitCost = crocoddyl.CostModelResidual(state, xLimitActivation, xLimitResidual)

# Cost for state and control
xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv) ** 2
)
uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
xTActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [100] * state.nv) ** 2
)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

# Cost for target reaching: hand and foot
handTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state, endEffectorId, pinocchio.SE3(np.eye(3), target), actuation.nu
)
handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1] * 3 + [0.0001] * 3) ** 2
)
handTrackingCost = crocoddyl.CostModelResidual(
    state, handTrackingActivation, handTrackingResidual
)

footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state, leftFootId, pinocchio.SE3(np.eye(3), np.array([0.0, 0.4, 0.0])), actuation.nu
)
footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1, 1, 0.1] + [1.0] * 3) ** 2
)
footTrackingCost1 = crocoddyl.CostModelResidual(
    state, footTrackingActivation, footTrackingResidual
)
footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    leftFootId,
    pinocchio.SE3(np.eye(3), np.array([0.3, 0.15, 0.35])),
    actuation.nu,
)
footTrackingCost2 = crocoddyl.CostModelResidual(
    state, footTrackingActivation, footTrackingResidual
)

# Cost for CoM reference
comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
comTrack = crocoddyl.CostModelResidual(state, comResidual)

# Create cost model per each action model. We divide the motion in 3 phases plus its
# terminal model.
runningCostModel1 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel2 = crocoddyl.CostModelSum(state, actuation.nu)
runningCostModel3 = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

# Then let's added the running and terminal cost functions
runningCostModel1.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel1.addCost("stateReg", xRegCost, 1e-3)
runningCostModel1.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel1.addCost("limitCost", limitCost, 1e3)

runningCostModel2.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel2.addCost("footPose", footTrackingCost1, 1e1)
runningCostModel2.addCost("stateReg", xRegCost, 1e-3)
runningCostModel2.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel2.addCost("limitCost", limitCost, 1e3)

runningCostModel3.addCost("gripperPose", handTrackingCost, 1e2)
runningCostModel3.addCost("footPose", footTrackingCost2, 1e1)
runningCostModel3.addCost("stateReg", xRegCost, 1e-3)
runningCostModel3.addCost("ctrlReg", uRegCost, 1e-4)
runningCostModel3.addCost("limitCost", limitCost, 1e3)

terminalCostModel.addCost("gripperPose", handTrackingCost, 1e2)
terminalCostModel.addCost("stateReg", xRegTermCost, 1e-3)
terminalCostModel.addCost("limitCost", limitCost, 1e3)


constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)

FORCE_COST_AND_CONSTRAINT = False
if FORCE_COST_AND_CONSTRAINT:
    fref = pinocchio.Force.Zero()
    ForceResidual = crocoddyl.ResidualModelContactForce(state, rightFootId, fref, 6, actuation.nu)
    constraintFriction = crocoddyl.ConstraintModelResidual(state, ForceResidual, np.array([0., 0, 0]*2), np.array([np.inf, np.inf, np.inf]*2))
    constraintModelManager.addConstraint("force", constraintFriction)

    Forcecost = crocoddyl.CostModelResidual(state, ForceResidual)
    runningCostModel1.addCost("forcecost1", Forcecost, 1e-3)
    runningCostModel2.addCost("forcecost2", Forcecost, 1e-3)
    runningCostModel3.addCost("forcecost3", Forcecost, 1e-3)

# Create the action model
dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel2Feet, runningCostModel1, constraintModelManager, 0., True)

dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel1Foot, runningCostModel2, constraintModelManager, 0., True)

dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel1Foot, runningCostModel3, constraintModelManager, 0., True)

dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel1Foot, terminalCostModel
)

runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

# Problem definition
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
problem = crocoddyl.ShootingProblem(
    x0, [runningModel1] * T + [runningModel2] * T + [runningModel3] * T, terminalModel
)

N_ocp = 3 * T


xs_init = [x0] * (N_ocp + 1)
us_init = [np.zeros( actuation.nu)] * N_ocp

LINE_WIDTH = 100

print(" TEST OSQP ".center(LINE_WIDTH, "-"))


ddp1 = mim_solvers.SolverCSQP(problem)
ddp2 = CSQP(problem, "OSQP")
ddp3 = CSQP(problem, "ProxQP")

ddp1.with_callbacks = False
ddp2.with_callbacks = False
ddp3.with_callbacks = False

max_qp_iters = 25
ddp1.max_qp_iters = max_qp_iters
ddp2.max_qp_iters = max_qp_iters
ddp3.max_qp_iters = max_qp_iters

eps_abs = 1e-20
eps_rel = 0.
ddp1.eps_abs = eps_abs
ddp1.eps_rel = eps_rel
ddp2.eps_abs = eps_abs
ddp2.eps_rel = eps_rel
ddp3.eps_abs = eps_abs
ddp3.eps_rel = eps_rel

ddp1.equality_qp_initial_guess = False
ddp2.equality_qp_initial_guess = False
ddp3.equality_qp_initial_guess = False

ddp1.update_rho_with_heuristic = True

# converged = ddp1.solve(xs_init, us_init, 1)
import time 



# t0 = time.time()
# ddp1.calc(True)
# tcalc =  time.time() - t0
# print("time tcalc = ", tcalc)


# t1 = time.time()
# converged = ddp1.solve(xs_init, us_init, 1)
# print("Stagewise time : ", time.time() - t1)
# print("Stagewise time minus cals : ", time.time() - t1 - tcalc)


converged = ddp1.solve(xs_init, us_init, 0)


t1 = time.time()
ddp1.computeDirection(True)
print("Stagewise computeDirection : ", time.time() - t1)

converged = ddp2.solve(xs_init, us_init, 1)
# converged = ddp3.solve(xs_init, us_init, 1)




print("Stgewise iter = ", ddp1.qp_iters)
print("OSQP iter = ", ddp2.qp_iters)
# print("Proxqp iter = ", ddp3.qp_iters)