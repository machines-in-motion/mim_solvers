///////////////////////////////////////////////////////////////////////////////
// 
// This file is a modified version of the solvers unittests from the Crocoddyl library
// This modified version is used for testing purposes only
// Original file : https://github.com/loco-3d/crocoddyl/blob/devel/unittest/test_solvers.cpp
// 
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <crocoddyl/core/utils/callbacks.hpp>
#include "factory/solver.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace mim_solvers::unittest;

//____________________________________________________________________________//

void test_solver_solutions(ProblemTypes::Type problem_type,
                           ModelTypes::Type model_type,
                           XConstraintType::Type x_cstr_type,
                           UConstraintType::Type u_cstr_type) {
  
  const double TOL = 1e-6;
  SolverFactory factory;
  boost::shared_ptr<crocoddyl::SolverAbstract> solverCSQP = factory.create(SolverTypes::SolverCSQP, problem_type, model_type, x_cstr_type, u_cstr_type);
  if(x_cstr_type == XConstraintType::None && u_cstr_type == UConstraintType::None){
    boost::shared_ptr<crocoddyl::SolverAbstract> solverSQP = factory.create(SolverTypes::SolverSQP, problem_type, model_type, x_cstr_type, u_cstr_type);
    solverCSQP->solve();
    solverSQP->solve();
    for(std::size_t t=0; t<solverCSQP->get_problem()->get_T(); t++){
      BOOST_CHECK((solverCSQP->get_us()[t] - solverSQP->get_us()[t]).isZero(TOL));
      BOOST_CHECK((solverCSQP->get_xs()[t] - solverSQP->get_xs()[t]).isZero(TOL));
    }
  }

  // // define some aliases
  // const std::size_t ndx = kkt->get_ndx();
  // const std::size_t nu = kkt->get_nu();

  // // Test the different matrix sizes
  // BOOST_CHECK_EQUAL(kkt->get_kkt().rows(), 2 * ndx + nu);
  // BOOST_CHECK_EQUAL(kkt->get_kkt().cols(), 2 * ndx + nu);
  // BOOST_CHECK_EQUAL(kkt->get_kktref().size(), 2 * ndx + nu);
  // BOOST_CHECK_EQUAL(kkt->get_primaldual().size(), 2 * ndx + nu);
  // BOOST_CHECK_EQUAL(kkt->get_us().size(), T);
  // BOOST_CHECK_EQUAL(kkt->get_xs().size(), T + 1);
}

//____________________________________________________________________________//

// void test_kkt_search_direction(ActionModelTypes::Type action_type, size_t T) {
//   // Create the kkt solver
//   SolverFactory factory;
//   boost::shared_ptr<mim_solvers::SolverKKT> kkt =
//       boost::static_pointer_cast<mim_solvers::SolverKKT>(
//           factory.create(SolverTypes::SolverKKT, action_type, T));

//   // Generate the different state along the trajectory
//   const boost::shared_ptr<crocoddyl::ShootingProblem>& problem =
//       kkt->get_problem();
//   const boost::shared_ptr<crocoddyl::StateAbstract>& state =
//       problem->get_runningModels()[0]->get_state();
//   std::vector<Eigen::VectorXd> xs;
//   std::vector<Eigen::VectorXd> us;
//   for (std::size_t i = 0; i < T; ++i) {
//     const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
//         problem->get_runningModels()[i];
//     xs.push_back(state->rand());
//     us.push_back(Eigen::VectorXd::Random(model->get_nu()));
//   }
//   xs.push_back(state->rand());

//   // Compute the search direction
//   kkt->setCandidate(xs, us);
//   kkt->computeDirection();

//   // define some aliases
//   const std::size_t ndx = kkt->get_ndx();
//   const std::size_t nu = kkt->get_nu();
//   Eigen::MatrixXd kkt_mat = kkt->get_kkt();
//   Eigen::Block<Eigen::MatrixXd> hess = kkt_mat.block(0, 0, ndx + nu, ndx + nu);

//   // Checking the symmetricity of the Hessian
//   double TOL = 1e-7;
//   BOOST_CHECK((hess - hess.transpose()).isZero(TOL));

//   // Check initial state
//   BOOST_CHECK((state->diff_dx(state->integrate_x(xs[0], kkt->get_dxs()[0]),
//                               kkt->get_problem()->get_x0()))
//                   .isZero(TOL));
// }

// //____________________________________________________________________________//

// void test_solver_against_kkt_solver(SolverTypes::Type solver_type,
//                                     ActionModelTypes::Type action_type,
//                                     size_t T) {
//   // Create the testing and KKT solvers
//   SolverFactory solver_factory;
//   boost::shared_ptr<crocoddyl::SolverAbstract> solver =
//       solver_factory.create(solver_type, action_type, T);
//   boost::shared_ptr<crocoddyl::SolverAbstract> kkt =
//       solver_factory.create(SolverTypes::SolverKKT, action_type, T);

//   // Get the pointer to the problem so we can create the equivalent kkt solver.
//   const boost::shared_ptr<crocoddyl::ShootingProblem>& problem =
//       solver->get_problem();

//   // Generate the different state along the trajectory
//   const boost::shared_ptr<crocoddyl::StateAbstract>& state =
//       problem->get_runningModels()[0]->get_state();
//   std::vector<Eigen::VectorXd> xs;
//   std::vector<Eigen::VectorXd> us;
//   for (std::size_t i = 0; i < T; ++i) {
//     const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
//         problem->get_runningModels()[i];
//     xs.push_back(state->rand());
//     us.push_back(Eigen::VectorXd::Random(model->get_nu()));
//   }
//   xs.push_back(state->rand());

//   // Define the callback function
//   std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract> > cbs;
//   cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
//   kkt->setCallbacks(cbs);
//   solver->setCallbacks(cbs);

//   // Print the name of the action model for introspection
//   std::cout << ActionModelTypes::all[action_type] << std::endl;
//   // std::cout << " - - - - - - - - - - - - - - - - - -" << std::endl;
//   std::cout << solver_type << " " << action_type << std::endl;

//   // Solve the problem using the KKT solver
//   // std::cout << " KKT solve..." << std::endl;
//   kkt->solve(xs, us, 100);

//   // Solve the problem using the solver in testing
//   // std::cout << " SOLVER solve..." << std::endl;
//   solver->solve(xs, us, 100);
//   // std::cout << " - - - - - - - - - - - - - - - - - -" << std::endl;

//   // check trajectory dimensions
//   BOOST_CHECK_EQUAL(solver->get_us().size(), T);
//   BOOST_CHECK_EQUAL(solver->get_xs().size(), T + 1);

//   // initial state
//   double TOL = 1e-4;
//   BOOST_CHECK((solver->get_xs()[0] - problem->get_x0()).isZero(TOL));

//   // check solutions against each other
//   for (unsigned int t = 0; t < T; ++t) {
//     const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
//         solver->get_problem()->get_runningModels()[t];
//     std::size_t nu = model->get_nu();
//     BOOST_CHECK((state->diff_dx(solver->get_xs()[t], kkt->get_xs()[t])).isZero(TOL));
//     BOOST_CHECK((solver->get_us()[t].head(nu) - kkt->get_us()[t]).isZero(TOL));
//   }
//   BOOST_CHECK(
//       (state->diff_dx(solver->get_xs()[T], kkt->get_xs()[T])).isZero(TOL));
//   // std::cout << "diff = " << std::endl << state->diff_dx(solver->get_xs()[T], kkt->get_xs()[T]) << std::endl;
// }

//____________________________________________________________________________//

void register_solver_unittest(ProblemTypes::Type problem_type,
                              ModelTypes::Type model_type,
                              XConstraintType::Type x_cstr_type,
                              UConstraintType::Type u_cstr_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << problem_type << "_" << model_type << "_" << x_cstr_type << "_" << u_cstr_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_solver_solutions, problem_type, model_type, x_cstr_type, u_cstr_type)));
  framework::master_test_suite().add(ts);
}

//____________________________________________________________________________//

bool init_function() {

  for (size_t i_pb = 0; i_pb < ProblemTypes::all.size(); ++i_pb) {
    for (size_t i_md = 0; i_md < ModelTypes::all.size(); ++i_md) {
      for (size_t i_xc = 0; i_xc < XConstraintType::all.size(); ++i_xc) {
        for (size_t i_uc = 0; i_uc < UConstraintType::all.size(); ++i_uc) {
          register_solver_unittest(ProblemTypes::all[i_pb],
                                   ModelTypes::all[i_md],
                                   XConstraintType::all[i_xc],
                                   UConstraintType::all[i_uc]);
        }
      }
    }
  }
  
  return true;

}

//____________________________________________________________________________//

int main(int argc, char* argv[]) {
  return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
