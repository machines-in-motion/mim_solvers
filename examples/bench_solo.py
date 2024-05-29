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




import crocoddyl
import pinocchio
import numpy as np
import example_robot_data 
import pinocchio as pin
import matplotlib.pyplot as plt


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
        data.r[0] = np.array([self.mu * F[2] - np.sqrt(F[0]**2 + F[1]**2)])

    def calcDiff(self, data, x, u=None):
        F = data.shared.contacts.contacts[self.contact_name].f.vector[:3]

        self.dcone_df[0, 0] = -F[0] / np.sqrt(F[0]**2 + F[1]**2)
        self.dcone_df[0, 1] = -F[1] / np.sqrt(F[0]**2 + F[1]**2)
        self.dcone_df[0, 2] = self.mu

        self.df_dx = data.shared.contacts.contacts[self.contact_name].df_dx[:3]   
        self.df_du = data.shared.contacts.contacts[self.contact_name].df_du[:3] 

        data.Rx = self.dcone_df @ self.df_dx 
        data.Ru = self.dcone_df @ self.df_du



pinRef        = pin.LOCAL_WORLD_ALIGNED
FRICTION_CSTR = True
MU = 0.8     # friction coefficient

PLOT_OCP_SOL = False
PLAY_OCP_SOL = True

robot_name = 'solo12'
ee_frame_names = ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
solo12 = example_robot_data.ROBOTS[robot_name]()
rmodel = solo12.robot.model
rmodel.type = 'QUADRUPED'
rmodel.foot_type = 'POINT_FOOT'
rdata = rmodel.createData()

# set contact frame_names and_indices
lfFootId = rmodel.getFrameId(ee_frame_names[0])
rfFootId = rmodel.getFrameId(ee_frame_names[1])
lhFootId = rmodel.getFrameId(ee_frame_names[2])
rhFootId = rmodel.getFrameId(ee_frame_names[3])


q0 = np.array([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0] 
                + 2 * [0.0, 0.8, -1.6] 
                + 2 * [0.0, -0.8, 1.6] 
                )

x0 =  np.concatenate([q0, np.zeros(rmodel.nv)])

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
for t in range(N_ocp+1):
    comDes_t = comRef.copy()
    w = (2 * np.pi) * 0.2 
    comDes_t[0] += radius * np.sin(w * t * dt) 
    comDes_t[1] += radius * (np.cos(w * t * dt) - 1)
    comDes += [comDes_t]

running_models = []
constraintModels = []
for t in range(N_ocp+1):
    contactModel = crocoddyl.ContactModelMultiple(state, nu)
    costModel = crocoddyl.CostModelSum(state, nu)

    # Add contact
    for frame_idx in supportFeetIds:
        support_contact = crocoddyl.ContactModel3D(state, frame_idx, np.array([0., 0., 0.]), pinRef, nu, np.array([0., 0.]))
        contactModel.addContact(rmodel.frames[frame_idx].name + "_contact", support_contact) 

    # Add state/control reg costs

    state_reg_weight, control_reg_weight = 1e-1, 1e-3

    freeFlyerQWeight = [0.]*3 + [500.]*3
    freeFlyerVWeight = [10.]*6
    legsQWeight = [0.01]*(rmodel.nv - 6)
    legsWWeights = [1.]*(rmodel.nv - 6)
    stateWeights = np.array(freeFlyerQWeight + legsQWeight + freeFlyerVWeight + legsWWeights)    


    stateResidual = crocoddyl.ResidualModelState(state, x0, nu)
    stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
    stateReg = crocoddyl.CostModelResidual(state, stateActivation, stateResidual)

    if t == N_ocp:
        costModel.addCost("stateReg", stateReg, state_reg_weight*dt)
    else:
        costModel.addCost("stateReg", stateReg, state_reg_weight)

    if t != N_ocp:
        ctrlResidual = crocoddyl.ResidualModelControl(state, nu)
        ctrlReg = crocoddyl.CostModelResidual(state, ctrlResidual)
        costModel.addCost("ctrlReg", ctrlReg, control_reg_weight)      
        
        FORCE_COST = True
        if FORCE_COST:
            for frame_idx in supportFeetIds:
                fref = pin.Force.Zero()# np.zeros(3)
                # fref[2] = 1
                ForceResidual = crocoddyl.ResidualModelContactForce(state, frame_idx, fref, 3, nu)
                ForceRes = crocoddyl.CostModelResidual(state, ForceResidual)
                costModel.addCost("ForceRes" + str(frame_idx), ForceRes, 1.)     


    # Add COM task
    com_residual = crocoddyl.ResidualModelCoMPosition(state, comDes[t], nu)
    com_activation = crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 1.]))
    com_track = crocoddyl.CostModelResidual(state, com_activation, com_residual)
    if t == N_ocp:
        costModel.addCost("comTrack", com_track, 1e5)
    else:
        costModel.addCost("comTrack", com_track, 1e5)

    constraintModelManager = crocoddyl.ConstraintModelManager(state, actuation.nu)
    if(FRICTION_CSTR):
        if(t != N_ocp):
            for frame_idx in supportFeetIds:
                name = rmodel.frames[frame_idx].name + "_contact"
                residualFriction = ResidualFrictionCone(state, name, MU, actuation.nu)
                constraintFriction = crocoddyl.ConstraintModelResidual(state, residualFriction, np.array([0.]), np.array([np.inf]))
                constraintModelManager.addConstraint(name + "friction", constraintFriction)

    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, contactModel, costModel, constraintModelManager, 0., True)
    model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)

    running_models += [model]

# Create shooting problem
problem = crocoddyl.ShootingProblem(x0, running_models[:-1], running_models[-1])


# # # # # # # # # # # # #
###     SOLVE OCP     ###
# # # # # # # # # # # # #

# Define warm start
xs_init = [x0] * (N_ocp + 1)
us_init = [np.zeros(nu)] * N_ocp

LINE_WIDTH = 100

print(" TEST OSQP ".center(LINE_WIDTH, "-"))


ddp1 = mim_solvers.SolverCSQP(problem)
ddp2 = CSQP(problem, "OSQP")
ddp3 = CSQP(problem, "ProxQP")
ddp4 = CSQP(problem, "HPIPM")

ddp1.with_callbacks = False
ddp2.with_callbacks = False
ddp3.with_callbacks = False
ddp4.with_callbacks = False

max_qp_iters = 25
ddp1.max_qp_iters = max_qp_iters
ddp2.max_qp_iters = max_qp_iters
ddp3.max_qp_iters = max_qp_iters
ddp4.max_qp_iters = max_qp_iters

eps_abs = 1e-20
eps_rel = 0.
ddp1.eps_abs = eps_abs
ddp1.eps_rel = eps_rel
ddp2.eps_abs = eps_abs
ddp2.eps_rel = eps_rel
ddp3.eps_abs = eps_abs
ddp3.eps_rel = eps_rel
ddp4.eps_abs = eps_abs
ddp4.eps_rel = eps_rel


ddp1.equality_qp_initial_guess = False
ddp2.equality_qp_initial_guess = False
ddp3.equality_qp_initial_guess = False
ddp4.equality_qp_initial_guess = False

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
converged = ddp4.solve(xs_init, us_init, 1)




print("Stagewise iter = ", ddp1.qp_iters)
print("OSQP iter = ", ddp2.qp_iters)
# print("Proxqp iter = ", ddp3.qp_iters)