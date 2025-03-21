"""
License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import numpy as np
import pinocchio as pin

try:
    import meshcat.geometry as g
    import meshcat.transformations as tf
except Exception:
    print("You need to install meshcat.")

import crocoddyl


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


def get_solution_trajectories(solver, rmodel, rdata, supportFeetIds):
    xs, us = solver.xs, solver.us
    nq, _, N = rmodel.nq, rmodel.nv, len(xs)
    jointPos_sol = []
    jointVel_sol = []
    jointAcc_sol = []
    jointTorques_sol = []
    centroidal_sol = []

    x = []
    for time_idx in range(N):
        q, v = xs[time_idx][:nq], xs[time_idx][nq:]
        pin.framesForwardKinematics(rmodel, rdata, q)
        pin.computeCentroidalMomentum(rmodel, rdata, q, v)
        centroidal_sol += [
            np.concatenate([pin.centerOfMass(rmodel, rdata, q, v), np.array(rdata.hg)])
        ]
        jointPos_sol += [q]
        jointVel_sol += [v]
        x += [xs[time_idx]]
        if time_idx < N - 1:
            jointAcc_sol += [solver.problem.runningDatas[time_idx].xnext[nq::]]
            jointTorques_sol += [us[time_idx]]

    sol = {
        "x": x,
        "centroidal": centroidal_sol,
        "jointPos": jointPos_sol,
        "jointVel": jointVel_sol,
        "jointAcc": jointAcc_sol,
        "jointTorques": jointTorques_sol,
    }

    for frame_idx in supportFeetIds:
        ct_frame_name = rmodel.frames[frame_idx].name + "_contact"
        datas = [
            solver.problem.runningDatas[i].differential.multibody.contacts.contacts[
                ct_frame_name
            ]
            for i in range(N - 1)
        ]
        ee_forces = [datas[k].f.vector for k in range(N - 1)]
        sol[ct_frame_name] = [ee_forces[i] for i in range(N - 1)]

    return sol


class Arrow(object):
    def __init__(
        self,
        meshcat_vis,
        name,
        location=[0, 0, 0],
        vector=[0, 0, 1],
        length_scale=1,
        color=0xFF0000,
    ):
        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.line = self.vis["line"]
        self.material = g.MeshBasicMaterial(color=color, reflectivity=0.5)

        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)

    def _update(self):
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length / 2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)

    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length / 0.08
        self.line.set_object(
            g.Cylinder(height=self.length, radius=0.005), self.material
        )
        self.cone.set_object(
            g.Cylinder(height=0.015, radius=0.01, radiusTop=0.0, radiusBottom=0.01),
            self.material,
        )
        self.cone.set_transform(tf.translation_matrix([0.0, cone_scale * 0.04, 0]))
        if update:
            self._update()

    def set_direction(self, direction, update=True):
        orientation = np.eye(4)
        orientation[:3, 0] = np.cross([1, 0, 0], direction)
        orientation[:3, 1] = direction
        orientation[:3, 2] = np.cross(orientation[:3, 0], orientation[:3, 1])
        self.orientation = orientation
        if update:
            self._update()

    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()

    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(np.array(vector) / np.linalg.norm(vector), False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()


class Cone(object):
    def __init__(
        self,
        meshcat_vis,
        name,
        location=[0, 0, 0],
        mu=1,
        vector=[0, 0, 1],
        length_scale=0.06,
    ):
        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.material = g.MeshBasicMaterial(
            color=0xFFFFFF, opacity=0.5, reflectivity=0.5
        )

        self.mu = mu * length_scale
        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)

    def _update(self):
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length / 2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)

    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length
        self.cone.set_object(
            g.Cylinder(
                height=cone_scale, radius=self.mu, radiusTop=self.mu, radiusBottom=0
            ),
            self.material,
        )
        if update:
            self._update()

    def set_direction(self, direction, update=True):
        orientation = np.eye(4)
        orientation[:3, 0] = np.cross([1, 0, 0], direction)
        orientation[:3, 1] = direction
        orientation[:3, 2] = np.cross(orientation[:3, 0], orientation[:3, 1])
        self.orientation = orientation
        if update:
            self._update()

    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()

    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(np.array(vector) / np.linalg.norm(vector), False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()
