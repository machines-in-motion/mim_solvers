"""__init__
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import sys

import crocoddyl
import example_robot_data
import numpy as np
import pinocchio as pin
from crocoddyl.utils.pendulum import (
    ActuationModelDoublePendulum,
    CostModelDoublePendulum,
)


class DiffActionModelCLQR(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isInitial=False, isTerminal=False):
        self.nq = 2
        self.nv = 2
        self.ndx = 2
        self.nx = self.nq + self.nv
        nu = 2
        nr = 1
        ng = 4
        nh = 0
        self.isTerminal = isTerminal
        self.isInitial = isInitial

        if self.isInitial:
            ng = 0

        state = crocoddyl.StateVector(self.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, nr, ng, nh)

        if not self.isInitial:
            lower_bound = np.array([-np.inf] * ng)
            upper_bound = np.array([0.4, 0.2, np.inf, np.inf])

            self.g_lb = lower_bound
            self.g_ub = upper_bound

        self.g = np.array([0.0, -9.81])

        self.x_weights_terminal = [200, 200, 10, 10]

    def _running_cost(self, x, u):
        cost = (x[0] - 1.0) ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        cost += u[0] ** 2 + u[1] ** 2
        return 0.5 * cost

    def _terminal_cost(self, x, u):
        cost = (
            self.x_weights_terminal[0] * ((x[0] - 1.0) ** 2)
            + self.x_weights_terminal[1] * (x[1] ** 2)
            + self.x_weights_terminal[2] * (x[2] ** 2)
            + self.x_weights_terminal[3] * (x[3] ** 2)
        )
        return 0.5 * cost

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xnext = np.zeros(self.state.nx)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = u + self.g

        if not self.isInitial:
            data.g = x.copy()

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros((2, 4))
        Fu = np.eye(2)

        Lx = np.zeros([4])
        Lu = np.zeros([2])
        Lxx = np.zeros([4, 4])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([4, 2])
        if self.isTerminal:
            Lx[0] = self.x_weights_terminal[0] * (x[0] - 1)
            Lx[1] = self.x_weights_terminal[1] * x[1]
            Lx[2] = self.x_weights_terminal[2] * x[2]
            Lx[3] = self.x_weights_terminal[3] * x[3]
            Lxx[0, 0] = self.x_weights_terminal[0]
            Lxx[1, 1] = self.x_weights_terminal[1]
            Lxx[2, 2] = self.x_weights_terminal[2]
            Lxx[3, 3] = self.x_weights_terminal[3]
        else:
            Lx[0] = x[0] - 1
            Lx[1] = x[1]
            Lx[2] = x[2]
            Lx[3] = x[3]
            Lu[0] = u[0]
            Lu[1] = u[1]
            Lxx = np.eye(4)
            Luu = np.eye(2)

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()

        if not self.isInitial:
            data.Gx = np.eye(self.nx)
            data.Gu = np.zeros((self.nx, self.nu))


def create_clqr_problem():
    diff_clqr_initial = DiffActionModelCLQR(isInitial=True)
    diff_clqr_running = DiffActionModelCLQR()
    diff_clqr_terminal = DiffActionModelCLQR(isTerminal=True)

    dt = 0.5
    clqr_initial = crocoddyl.IntegratedActionModelEuler(diff_clqr_initial, dt)
    clqr_running = crocoddyl.IntegratedActionModelEuler(diff_clqr_running, dt)
    clqr_terminal = crocoddyl.IntegratedActionModelEuler(diff_clqr_terminal, 0.0)

    horizon = 20
    x0 = np.zeros(4)
    problem = crocoddyl.ShootingProblem(
        x0, [clqr_initial] + [clqr_running] * (horizon - 1), clqr_terminal
    )

    nx = 4
    nu = 2

    xs_init = [x0] + [10 * np.ones(nx)] * (horizon)
    us_init = [np.ones(nu) for t in range(horizon)]
    return problem, xs_init, us_init


class DiffActionModelLQR(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isTerminal=False):
        self.nq = 2
        self.nv = 2
        self.ndx = 2
        self.nx = self.nq + self.nv
        nu = 2
        nr = 1
        ng = 0
        nh = 0
        self.isTerminal = isTerminal

        state = crocoddyl.StateVector(self.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, nr, ng, nh)

        self.g = np.array([0.0, -9.81])

        self.x_weights_terminal = [200, 200, 10, 10]

    def _running_cost(self, x, u):
        cost = (x[0] - 1.0) ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
        cost += u[0] ** 2 + u[1] ** 2
        return 0.5 * cost

    def _terminal_cost(self, x, u):
        cost = (
            self.x_weights_terminal[0] * ((x[0] - 1.0) ** 2)
            + self.x_weights_terminal[1] * (x[1] ** 2)
            + self.x_weights_terminal[2] * (x[2] ** 2)
            + self.x_weights_terminal[3] * (x[3] ** 2)
        )
        return 0.5 * cost

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xnext = np.zeros(self.state.nx)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = u + self.g

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros((2, 4))
        Fu = np.eye(2)

        Lx = np.zeros([4])
        Lu = np.zeros([2])
        Lxx = np.zeros([4, 4])
        Luu = np.zeros([2, 2])
        Lxu = np.zeros([4, 2])
        if self.isTerminal:
            Lx[0] = self.x_weights_terminal[0] * (x[0] - 1)
            Lx[1] = self.x_weights_terminal[1] * x[1]
            Lx[2] = self.x_weights_terminal[2] * x[2]
            Lx[3] = self.x_weights_terminal[3] * x[3]
            Lxx[0, 0] = self.x_weights_terminal[0]
            Lxx[1, 1] = self.x_weights_terminal[1]
            Lxx[2, 2] = self.x_weights_terminal[2]
            Lxx[3, 3] = self.x_weights_terminal[3]
        else:
            Lx[0] = x[0] - 1
            Lx[1] = x[1]
            Lx[2] = x[2]
            Lx[3] = x[3]
            Lu[0] = u[0]
            Lu[1] = u[1]
            Lxx = np.eye(4)
            Luu = np.eye(2)

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


def create_lqr_problem():
    lq_diff_running = DiffActionModelLQR()
    lq_diff_terminal = DiffActionModelLQR(isTerminal=True)
    dt = 0.1
    horizon = 100
    x0 = np.zeros(4)
    lq_running = crocoddyl.IntegratedActionModelEuler(lq_diff_running, dt)
    lq_terminal = crocoddyl.IntegratedActionModelEuler(lq_diff_terminal, dt)

    problem = crocoddyl.ShootingProblem(x0, [lq_running] * horizon, lq_terminal)

    xs_init = [10 * np.ones(4)] * (horizon + 1)
    us_init = [np.ones(2) * 1 for t in range(horizon)]
    return problem, xs_init, us_init


def create_double_pendulum_problem():
    """
    Create shooting problem for the double pendulum model
    """
    print("Created double pendulum problem ...")
    # Loading the double pendulum model
    pendulum = example_robot_data.load("double_pendulum")
    model = pendulum.model
    state = crocoddyl.StateMultibody(model)
    actuation = ActuationModelDoublePendulum(state, actLink=1)
    nu = actuation.nu
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xPendCost = CostModelDoublePendulum(
        state,
        crocoddyl.ActivationModelWeightedQuad(np.array([1.0] * 4 + [0.1] * 2)),
        nu,
    )
    dt = 1e-2
    runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
    runningCostModel.addCost("xGoal", xPendCost, 5e-1 / dt)
    terminalCostModel.addCost("xGoal", xPendCost, 10.0)
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        ),
        dt,
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        dt,
    )
    T = 100
    x0 = np.array([3.14, 0.0, 0.1, 0.0])
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    xs_init = [x0] * (problem.T + 1)
    us_init = problem.quasiStatic([x0] * problem.T)

    return problem, xs_init, us_init


def create_quadrotor_problem():
    """
    Create shooting problem for quadrotor task
    """
    print("Create quadrotor problem ...")
    hector = example_robot_data.load("hector")
    robot_model = hector.model
    target_pos = np.array([1.0, 0.0, 1.0])
    target_quat = pin.Quaternion(1.0, 0.0, 0.0, 0.0)
    state = crocoddyl.StateMultibody(robot_model)
    d_cog, cf, cm, _, _ = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
    tau_f = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, d_cog, 0.0, -d_cog],
            [-d_cog, 0.0, d_cog, 0.0],
            [-cm / cf, cm / cf, -cm / cf, cm / cf],
        ]
    )
    actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

    nu = actuation.nu
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)
    # Costs
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0.1] * 3 + [1000.0] * 3 + [1000.0] * robot_model.nv)
    )
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        robot_model.getFrameId("base_link"),
        pin.SE3(target_quat.matrix(), target_pos),
        nu,
    )
    goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
    runningCostModel.addCost("xReg", xRegCost, 1e-6)
    runningCostModel.addCost("uReg", uRegCost, 1e-6)
    runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
    terminalCostModel.addCost("goalPose", goalTrackingCost, 3.0)
    dt = 3e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        ),
        dt,
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        dt,
    )

    # Creating the shooting problem and the FDDP solver
    T = 33
    x0 = np.array(list(hector.q0) + [0.0] * hector.model.nv)
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    xs_init = [x0] * (problem.T + 1)
    us_init = problem.quasiStatic([x0] * problem.T)

    return problem, xs_init, us_init


def create_unconstrained_ur5():
    robot = example_robot_data.load("ur5")
    model = robot.model
    nv = model.nv
    nu = nv
    q0 = np.array([0, 0, 0, 0, 0, 0])
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0]).copy()

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

    # Define warm start
    xs_init = [x0] * (T + 1)
    us_init = [np.zeros(nu)] * T

    return problem, xs_init, us_init


def create_taichi():
    # Load robot
    robot = example_robot_data.load("talos")
    rmodel = robot.model
    lims = rmodel.effortLimit
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
    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacements(rmodel, rdata)
    rfPos0 = rdata.oMf[rightFootId].translation
    lfPos0 = rdata.oMf[leftFootId].translation
    comRef = (rfPos0 + lfPos0) / 2
    comRef[2] = pin.centerOfMass(rmodel, rdata, q0)[2].item()

    # Create two contact models used along the motion
    contactModel1Foot = crocoddyl.ContactModelMultiple(state, actuation.nu)
    contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
    supportContactModelLeft = crocoddyl.ContactModel6D(
        state,
        leftFootId,
        pin.SE3.Identity(),
        pin.LOCAL,
        actuation.nu,
        np.array([0, 40]),
    )
    supportContactModelRight = crocoddyl.ContactModel6D(
        state,
        rightFootId,
        pin.SE3.Identity(),
        pin.LOCAL,
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
        state, endEffectorId, pin.SE3(np.eye(3), target), actuation.nu
    )
    handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1] * 3 + [0.0001] * 3) ** 2
    )
    handTrackingCost = crocoddyl.CostModelResidual(
        state, handTrackingActivation, handTrackingResidual
    )

    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, leftFootId, pin.SE3(np.eye(3), np.array([0.0, 0.4, 0.0])), actuation.nu
    )
    footTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1, 1, 0.1] + [1.0] * 3) ** 2
    )
    footTrackingCost1 = crocoddyl.CostModelResidual(
        state, footTrackingActivation, footTrackingResidual
    )
    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, leftFootId, pin.SE3(np.eye(3), np.array([0.3, 0.15, 0.35])), actuation.nu
    )
    footTrackingCost2 = crocoddyl.CostModelResidual(
        state, footTrackingActivation, footTrackingResidual
    )

    # Cost for CoM reference
    # comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)

    # Create cost model per each action model. We divide the motion in 3 phases plus its terminal model
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

    # Create the action model
    dmodelRunning1 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel2Feet, runningCostModel1
    )
    dmodelRunning2 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, runningCostModel2
    )
    dmodelRunning3 = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, runningCostModel3
    )
    dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1Foot, terminalCostModel
    )

    runningModel1 = crocoddyl.IntegratedActionModelEuler(dmodelRunning1, DT)
    runningModel2 = crocoddyl.IntegratedActionModelEuler(dmodelRunning2, DT)
    runningModel3 = crocoddyl.IntegratedActionModelEuler(dmodelRunning3, DT)
    terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

    # Problem definition
    x0 = np.concatenate([q0, pin.utils.zero(state.nv)])
    problem = crocoddyl.ShootingProblem(
        x0,
        [runningModel1] * T + [runningModel2] * T + [runningModel3] * T,
        terminalModel,
    )

    # Warm-start
    # Solving it with the DDP algorithm
    xs_init = [x0] * (problem.T + 1)
    us_init = problem.quasiStatic([x0] * problem.T)
    return problem, xs_init, us_init
