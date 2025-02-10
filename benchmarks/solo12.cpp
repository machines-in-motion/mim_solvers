#include <iostream>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "math.h"
#include "mim_solvers/csqp.hpp"
#include "mim_solvers/sqp.hpp"
#include "timings.hpp"

int main() {
  // LOADING THE ROBOT AND INIT VARIABLES

  std::string urdf_path =
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/solo_description/robots/solo12.urdf";
  std::shared_ptr<pinocchio::Model> rmodel =
      std::make_shared<pinocchio::Model>();
  pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(),
                              *rmodel.get());

  // rmodel->type = "QUADRUPED";
  // rmodel->foot_type = "POINT_FOOT";
  pinocchio::Data rdata = pinocchio::Data(*rmodel.get());

  // set contact frame_names and_indices
  const int lfFootId = rmodel->getFrameId("FL_FOOT");
  const int rfFootId = rmodel->getFrameId("FR_FOOT");
  const int lhFootId = rmodel->getFrameId("HL_FOOT");
  const int rhFootId = rmodel->getFrameId("HR_FOOT");

  const int nq = rmodel->nq;
  const int nv = rmodel->nv;

  Eigen::VectorXd q0(nq);
  q0 << 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.8, -1.6, 0.0, 0.8, -1.6, 0.0,
      -0.8, 1.6, 0.0, -0.8, 1.6;
  Eigen::VectorXd v0 = Eigen::VectorXd::Zero(nv);

  Eigen::VectorXd x0;
  x0.resize(nq + nv);
  x0.head(nq) = q0;
  x0.tail(nv) = v0;

  pinocchio::forwardKinematics(*rmodel.get(), rdata, q0);
  pinocchio::updateFramePlacements(*rmodel.get(), rdata);

  Eigen::Vector3d rfFootPos0 = rdata.oMf[rfFootId].translation();
  Eigen::Vector3d rhFootPos0 = rdata.oMf[rhFootId].translation();
  Eigen::Vector3d lfFootPos0 = rdata.oMf[lfFootId].translation();
  Eigen::Vector3d lhFootPos0 = rdata.oMf[lhFootId].translation();

  std::vector<int> supportFeetIds = {lfFootId, rfFootId, lhFootId, rhFootId};

  // OCP Parameters
  const int N_ocp = 50;
  const double dt = 0.02;
  const double radius = 0.065;

  // OCP references for cost function
  Eigen::Vector3d comRef =
      0.25 * (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0);
  comRef[2] = pinocchio::centerOfMass(*rmodel.get(), rdata, q0)[2];

  std::vector<Eigen::Vector3d> comDes(N_ocp + 1, comRef);

  for (unsigned t = 0; t < N_ocp + 1; ++t) {
    const double w = 2 * M_PI * 0.2;
    comDes[t][0] += radius * sin(w * t * dt);
    comDes[t][1] += radius * (cos(w * t * dt) - 1);
  }

  // Crocoddyl variables
  std::shared_ptr<crocoddyl::StateMultibody> state =
      std::make_shared<crocoddyl::StateMultibody>(rmodel);
  std::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation =
      std::make_shared<crocoddyl::ActuationModelFloatingBase>(state);
  const int nu = actuation->get_nu();

  // CREATING RUNNING MODELS
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels;
  std::shared_ptr<crocoddyl::IntegratedActionModelEuler> terminal_model;

  const double state_reg_weight = 1e-1;
  const double control_reg_weight = 1e-3;

  Eigen::VectorXd stateWeights;
  stateWeights.resize(nv + nv);
  stateWeights.setZero();
  stateWeights.head(6) << 0.0, 0.0, 0.0, 500, 500, 500;
  stateWeights.segment(6, nv - 6) << (Eigen::VectorXd::Constant(nv - 6, 0.01));
  stateWeights.segment(nv, 6) << (Eigen::VectorXd::Constant(6, 10));
  stateWeights.segment(nv + 6, nv - 6)
      << (Eigen::VectorXd::Constant(nv - 6, 1.0));

  for (unsigned t = 0; t < N_ocp + 1; ++t) {
    std::shared_ptr<crocoddyl::ContactModelMultiple> contactModel =
        std::make_shared<crocoddyl::ContactModelMultiple>(state, nu);
    std::shared_ptr<crocoddyl::CostModelSum> costModel =
        std::make_shared<crocoddyl::CostModelSum>(state, nu);

    for (unsigned idx = 0; idx < supportFeetIds.size(); ++idx) {
      std::shared_ptr<crocoddyl::ContactModel3D> support_contact =
          std::make_shared<crocoddyl::ContactModel3D>(
              state, supportFeetIds[idx], Eigen::Vector3d::Zero(),
              pinocchio::LOCAL_WORLD_ALIGNED, nu, Eigen::Vector2d::Zero());
      contactModel->addContact(
          rmodel->frames[supportFeetIds[idx]].name + "_contact",
          support_contact);
    }
    std::shared_ptr<crocoddyl::ResidualModelState> stateResidual =
        std::make_shared<crocoddyl::ResidualModelState>(state, x0, nu);
    std::shared_ptr<crocoddyl::ActivationModelWeightedQuad> stateActivation =
        std::make_shared<crocoddyl::ActivationModelWeightedQuad>(
            stateWeights.array().square());
    std::shared_ptr<crocoddyl::CostModelResidual> stateReg =
        std::make_shared<crocoddyl::CostModelResidual>(state, stateActivation,
                                                       stateResidual);

    if (t == N_ocp) {
      costModel->addCost("stateReg", stateReg, state_reg_weight * dt);
    } else {
      costModel->addCost("stateReg", stateReg, state_reg_weight);
    }

    if (t != N_ocp) {
      std::shared_ptr<crocoddyl::ResidualModelControl> ctrlResidual =
          std::make_shared<crocoddyl::ResidualModelControl>(state, nu);
      std::shared_ptr<crocoddyl::CostModelResidual> ctrlReg =
          std::make_shared<crocoddyl::CostModelResidual>(state, ctrlResidual);
      costModel->addCost("ctrlReg", ctrlReg, control_reg_weight);
    }

    std::shared_ptr<crocoddyl::ResidualModelCoMPosition> com_residual =
        std::make_shared<crocoddyl::ResidualModelCoMPosition>(state, comDes[t],
                                                              nu);
    std::shared_ptr<crocoddyl::ActivationModelWeightedQuad> com_activation =
        std::make_shared<crocoddyl::ActivationModelWeightedQuad>(
            Eigen::Vector3d::Ones());
    std::shared_ptr<crocoddyl::CostModelResidual> com_track =
        std::make_shared<crocoddyl::CostModelResidual>(state, com_activation,
                                                       com_residual);

    costModel->addCost("comTrack", com_track, 1e5);

    std::shared_ptr<crocoddyl::ConstraintModelManager> constraints =
        std::make_shared<crocoddyl::ConstraintModelManager>(state, nu);

    std::shared_ptr<crocoddyl::ResidualModelState> stateResidualc =
        std::make_shared<crocoddyl::ResidualModelState>(state, x0, nu);

    Eigen::VectorXd x_lim;
    x_lim.resize(nq + nv);
    x_lim.head(nq + nv) << Eigen::VectorXd::Ones(nq + nv);

    std::shared_ptr<crocoddyl::ConstraintModelResidual> state_constraint =
        std::make_shared<crocoddyl::ConstraintModelResidual>(
            state, stateResidualc, x0 - x_lim, x0 + x_lim);
    constraints->addConstraint("State constraint", state_constraint);

    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>
        running_DAM = std::make_shared<
            crocoddyl::DifferentialActionModelContactFwdDynamics>(
            state, actuation, contactModel, costModel, constraints, 0.0, true);
    if (t != N_ocp) {
      std::shared_ptr<crocoddyl::IntegratedActionModelEuler> running_model =
          std::make_shared<crocoddyl::IntegratedActionModelEuler>(running_DAM,
                                                                  dt);
      runningModels.push_back(running_model);
    } else {
      terminal_model = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
          running_DAM, dt);
    }
  }

  std::shared_ptr<crocoddyl::ShootingProblem> problem =
      std::make_shared<crocoddyl::ShootingProblem>(x0, runningModels,
                                                   terminal_model);

  mim_solvers::Timer timer;

  std::cout << std::left << std::setw(42) << "      "
            << "  " << std::left << std::setw(15) << "AVG (ms)" << std::left
            << std::setw(15) << "STDDEV (ms)" << std::left << std::setw(15)
            << "MAX (ms)" << std::left << std::setw(15) << "MIN (ms)"
            << std::endl;

  // SETTING UP WARM START
  const std::size_t N = problem->get_T();
  std::vector<Eigen::VectorXd> xs(N, x0);
  std::vector<Eigen::VectorXd> us = problem->quasiStatic_xs(xs);
  xs.push_back(x0);

  // DEFINE SOLVER
  mim_solvers::SolverCSQP solver_CSQP = mim_solvers::SolverCSQP(problem);
  solver_CSQP.set_termination_tolerance(1e-4);
  // solver_CSQP.setCallbacks(false);
  solver_CSQP.set_eps_abs(0.0);
  solver_CSQP.set_eps_rel(0.0);
  solver_CSQP.set_max_qp_iters(50);
  solver_CSQP.set_equality_qp_initial_guess(false);

  // SETTING UP STATISTICS
  const int nb_CSQP = 1000;
  Eigen::VectorXd duration_CSQP(nb_CSQP);
  solver_CSQP.solve(xs, us, 0);
  for (unsigned i = 0; i < nb_CSQP; ++i) {
    timer.start();
    solver_CSQP.computeDirection(true);
    timer.stop();
    duration_CSQP[i] = timer.elapsed().user;
  }
  double const std_dev_CSQP =
      std::sqrt((duration_CSQP.array() - duration_CSQP.mean()).square().sum() /
                (nb_CSQP - 1));

  std::cout << "  " << std::left << std::setw(42) << "SOLO CSQP (50 QP iter)"
            << std::left << std::setw(15) << duration_CSQP.mean() << std::left
            << std::setw(15) << std_dev_CSQP << std::left << std::setw(15)
            << duration_CSQP.maxCoeff() << std::left << std::setw(15)
            << duration_CSQP.minCoeff() << std::endl;

  return 0;
}
