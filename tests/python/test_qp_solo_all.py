"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that the stagewise QP solver matches the other QP solvers implemented in python
on the solo CoM example (with constraints)
"""

import importlib.util
import os
import pathlib
import time

import crocoddyl
import example_robot_data
import mim_solvers
import numpy as np
import pinocchio
import pinocchio as pin

python_path = pathlib.Path(".").absolute().parent.parent / "python"
os.sys.path.insert(1, str(python_path))

from csqp import CSQP  # noqa: E402

HPIPM_PYTHON_FOUND = importlib.util.find_spec("hpipm_python")


class ResidualFrictionCone(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, contact_name, mu, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, nu, True, True, True)
        self.mu = mu
        self.contact_name = contact_name

        self.dcone_df = np.zeros((1, 3))
        self.df_dx = np.zeros((3, self.state.ndx))
        self.df_du = np.zeros((3, self.nu))

    def calc(self, data, x, u=None):
        F = data.shared.contacts.contacts[self.contact_name].f.vector[:3]
        data.r[0] = np.array([self.mu * F[2] - np.sqrt(F[0] ** 2 + F[1] ** 2)])

    def calcDiff(self, data, x, u=None):
        F = data.shared.contacts.contacts[self.contact_name].f.vector[:3]

        self.dcone_df[0, 0] = -F[0] / np.sqrt(F[0] ** 2 + F[1] ** 2)
        self.dcone_df[0, 1] = -F[1] / np.sqrt(F[0] ** 2 + F[1] ** 2)
        self.dcone_df[0, 2] = self.mu

        self.df_dx = data.shared.contacts.contacts[self.contact_name].df_dx[:3]
        self.df_du = data.shared.contacts.contacts[self.contact_name].df_du[:3]

        data.Rx = self.dcone_df @ self.df_dx
        data.Ru = self.dcone_df @ self.df_du


pinRef = pin.LOCAL_WORLD_ALIGNED
FRICTION_CSTR = True
MU = 0.8  # friction coefficient

PLOT_OCP_SOL = False
PLAY_OCP_SOL = True

robot_name = "solo12"
ee_frame_names = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
solo12 = example_robot_data.ROBOTS[robot_name]()
rmodel = solo12.robot.model
rmodel.type = "QUADRUPED"
rmodel.foot_type = "POINT_FOOT"
rdata = rmodel.createData()

# set contact frame_names and_indices
lfFootId = rmodel.getFrameId(ee_frame_names[0])
rfFootId = rmodel.getFrameId(ee_frame_names[1])
lhFootId = rmodel.getFrameId(ee_frame_names[2])
rhFootId = rmodel.getFrameId(ee_frame_names[3])


q0 = np.array(
    [0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0] + 2 * [0.0, 0.8, -1.6] + 2 * [0.0, -0.8, 1.6]
)

x0 = np.concatenate([q0, np.zeros(rmodel.nv)])

pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
rfFootPos0 = rdata.oMf[rfFootId].translation
rhFootPos0 = rdata.oMf[rhFootId].translation
lfFootPos0 = rdata.oMf[lfFootId].translation
lhFootPos0 = rdata.oMf[lhFootId].translation

comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()

supportFeetIds = [lfFootId, rfFootId, lhFootId, rhFootId]
supportFeePos = [lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0]


state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)
nu = actuation.nu


comDes = []

N_ocp = 100
dt = 0.02
T = N_ocp * dt
radius = 0.065
for t in range(N_ocp + 1):
    comDes_t = comRef.copy()
    w = (2 * np.pi) * 0.2
    comDes_t[0] += radius * np.sin(w * t * dt)
    comDes_t[1] += radius * (np.cos(w * t * dt) - 1)
    comDes += [comDes_t]

running_models = []
constraintModels = []
for t in range(N_ocp + 1):
    contactModel = crocoddyl.ContactModelMultiple(state, nu)
    costModel = crocoddyl.CostModelSum(state, nu)

    # Add contact
    for frame_idx in supportFeetIds:
        support_contact = crocoddyl.ContactModel3D(
            state,
            frame_idx,
            np.array([0.0, 0.0, 0.0]),
            pinRef,
            nu,
            np.array([0.0, 0.0]),
        )
        contactModel.addContact(
            rmodel.frames[frame_idx].name + "_contact", support_contact
        )

    # Add state/control reg costs

    state_reg_weight, control_reg_weight = 1e-1, 1e-3

    freeFlyerQWeight = [0.0] * 3 + [500.0] * 3
    freeFlyerVWeight = [10.0] * 6
    legsQWeight = [0.01] * (rmodel.nv - 6)
    legsWWeights = [1.0] * (rmodel.nv - 6)
    stateWeights = np.array(
        freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights
    )

    stateResidual = crocoddyl.ResidualModelState(state, x0, nu)
    stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
    stateReg = crocoddyl.CostModelResidual(state, stateActivation, stateResidual)

    if t == N_ocp:
        costModel.addCost("stateReg", stateReg, state_reg_weight * dt)
    else:
        costModel.addCost("stateReg", stateReg, state_reg_weight)

    if t != N_ocp:
        ctrlResidual = crocoddyl.ResidualModelControl(state, nu)
        ctrlReg = crocoddyl.CostModelResidual(state, ctrlResidual)
        costModel.addCost("ctrlReg", ctrlReg, control_reg_weight)

        FORCE_COST = True
        if FORCE_COST:
            for frame_idx in supportFeetIds:
                fref = pin.Force.Zero()  # np.zeros(3)
                # fref[2] = 1
                ForceResidual = crocoddyl.ResidualModelContactForce(
                    state, frame_idx, fref, 3, nu
                )
                ForceRes = crocoddyl.CostModelResidual(state, ForceResidual)
                costModel.addCost("ForceRes" + str(frame_idx), ForceRes, 1.0)

    # Add COM task
    com_residual = crocoddyl.ResidualModelCoMPosition(state, comDes[t], nu)
    com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 1.0]))
    com_track = crocoddyl.CostModelResidual(state, com_activation, com_residual)
    if t == N_ocp:
        costModel.addCost("comTrack", com_track, 1e5)
    else:
        costModel.addCost("comTrack", com_track, 1e5)

    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    if FRICTION_CSTR:
        if t != N_ocp:
            for frame_idx in supportFeetIds:
                name = rmodel.frames[frame_idx].name + "_contact"
                residualFriction = ResidualFrictionCone(state, name, MU, actuation.nu)
                constraintFriction = crocoddyl.ConstraintModelResidual(
                    state, residualFriction, np.array([0.0]), np.array([np.inf])
                )
                constraintModelManager.addConstraint(
                    name + "friction", constraintFriction
                )

    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel, costModel, constraintModelManager, 0.0, True
    )
    model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)

    running_models += [model]

# Create shooting problem
problem = crocoddyl.ShootingProblem(x0, running_models[:-1], running_models[-1])


# # # # # # # # # # # # #
###     SOLVE OCP     ###
# # # # # # # # # # # # #

# Define warm start
xs_init = [x0] * (N_ocp + 1)
us_init = problem.quasiStatic([problem.x0] * problem.T)

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
