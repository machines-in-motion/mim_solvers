"""
License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import sys

import crocoddyl
import example_robot_data
import matplotlib.pyplot as plt
import mim_solvers
import numpy as np
import pinocchio
import pinocchio as pin
import utils_solo12

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

N_ocp = 250
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
                residualFriction = utils_solo12.ResidualFrictionCone(
                    state, name, MU, actuation.nu
                )
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
ocp = crocoddyl.ShootingProblem(x0, running_models[:-1], running_models[-1])

solver = mim_solvers.SolverCSQP(ocp)
solver.max_qp_iters = 1000
max_iter = 500
solver.setCallbacks([mim_solvers.CallbackVerbose()])
solver.use_filter_line_search = False
solver.termination_tolerance = 1e-4
solver.mu_dynamic = -1  # use Nocedal's L1 merit based on Lagrange multipliers norm
solver.lag_mul_inf_norm_coef = 10.0
solver.eps_abs = 1e-6
solver.eps_rel = 1e-6


xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)

solver.setCallbacks([mim_solvers.CallbackVerbose(), mim_solvers.CallbackLogger()])

solver.solve(xs, us, max_iter)
solution = utils_solo12.get_solution_trajectories(solver, rmodel, rdata, supportFeetIds)

log = solver.getCallbacks()[-1]
crocoddyl.plotOCSolution(solver.xs, solver.us)
mim_solvers.plotConvergence(log.convergence_data)


# Plot solution of the constrained OCP
if PLOT_OCP_SOL:
    # Plot forces
    time_lin = np.linspace(0, T, solver.problem.T)
    fig, axs = plt.subplots(4, 3, constrained_layout=True)
    for i, frame_idx in enumerate(supportFeetIds):
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
        forces = np.array(solution[ct_frame_name])
        axs[i, 0].plot(time_lin, forces[:, 0], label="Fx")
        axs[i, 1].plot(time_lin, forces[:, 1], label="Fy")
        axs[i, 2].plot(time_lin, forces[:, 2], label="Fz")
        Fz_lb = (1.0 / MU) * np.sqrt(forces[:, 0] ** 2 + forces[:, 1] ** 2)
        axs[i, 2].plot(time_lin, Fz_lb, "k-.", label="lb")
        axs[i, 0].grid()
        axs[i, 1].grid()
        axs[i, 2].grid()
        axs[i, 0].set_ylabel(ct_frame_name)
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[0, 2].legend()

    axs[3, 0].set_xlabel(r"$F_x$")
    axs[3, 1].set_xlabel(r"$F_y$")
    axs[3, 2].set_xlabel(r"$F_z$")
    fig.suptitle("Force", fontsize=16)

    comDes = np.array(comDes)
    centroidal_sol = np.array(solution["centroidal"])
    plt.figure()
    plt.plot(comDes[:, 0], comDes[:, 1], "--", label="reference")
    plt.plot(centroidal_sol[:, 0], centroidal_sol[:, 1], label="solution")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("COM trajectory")
    plt.show()


# Animate solution tof the constrained OCP
if PLAY_OCP_SOL:
    robot = solo12.robot
    viz = pin.visualize.MeshcatVisualizer(
        robot.model, robot.collision_model, robot.visual_model
    )
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print(err)
        sys.exit(0)
    viz.loadViewerModel()

    arrow1 = utils_solo12.Arrow(
        viz.viewer,
        "force_1",
        location=[0, 0, 0],
        vector=[0, 0, 0.01],
        length_scale=0.05,
    )
    arrow2 = utils_solo12.Arrow(
        viz.viewer,
        "force_2",
        location=[0, 0, 0],
        vector=[0, 0, 0.01],
        length_scale=0.05,
    )
    arrow3 = utils_solo12.Arrow(
        viz.viewer,
        "force_3",
        location=[0, 0, 0],
        vector=[0, 0, 0.01],
        length_scale=0.05,
    )
    arrow4 = utils_solo12.Arrow(
        viz.viewer,
        "force_4",
        location=[0, 0, 0],
        vector=[0, 0, 0.01],
        length_scale=0.05,
    )

    cone1 = utils_solo12.Cone(
        viz.viewer, "friction_cone_1", location=supportFeePos[0], mu=MU
    )
    cone2 = utils_solo12.Cone(
        viz.viewer, "friction_cone_2", location=supportFeePos[1], mu=MU
    )
    cone3 = utils_solo12.Cone(
        viz.viewer, "friction_cone_3", location=supportFeePos[2], mu=MU
    )
    cone4 = utils_solo12.Cone(
        viz.viewer, "friction_cone_4", location=supportFeePos[3], mu=MU
    )

    arrows = [arrow1, arrow2, arrow3, arrow4]
    forces = []

    for i, contactLoc in enumerate(supportFeePos):
        ct_frame_name = rmodel.frames[supportFeetIds[i]].name + "_contact"
        forces.append(np.array(solution[ct_frame_name])[:, :3])
        arrows[i].set_location(contactLoc)

    import time

    viz.display(solution["jointPos"][0])
    time.sleep(2)
    for t in range(N_ocp):
        time.sleep(dt)
        viz.display(solution["jointPos"][t])

        for i in range(len(supportFeePos)):
            arrows[i].anchor_as_vector(supportFeePos[i], forces[i][t])
