///////////////////////////////////////////////////////////////////////////////
//
// This file is a modified version of the solvers unittests from the Crocoddyl
// library This modified version is used for testing purposes only Original file
// : https://github.com/loco-3d/crocoddyl/blob/devel/unittest/test_solvers.cpp
//
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include <crocoddyl/core/solver-base.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>

#include "factory/solver.hpp"
#include "mim_solvers/utils/callbacks.hpp"
#include "unittest_common.hpp"

using namespace boost::unit_test;
using namespace mim_solvers::unittest;

const double TOL = 1e-6;

//____________________________________________________________________________//

void test_sqp_core() {
  // Create solver
  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver = factory.create(
      SolverTypes::SolverSQP, ProblemTypes::ShootingProblem,
      ModelTypes::PointMass2D, XConstraintType::None, UConstraintType::None);
  // Downcast
  std::shared_ptr<mim_solvers::SolverSQP> solver_cast =
      std::static_pointer_cast<mim_solvers::SolverSQP>(solver);

  // Test initial & default attributes
  BOOST_CHECK_EQUAL(solver_cast->get_KKT(),
                    std::numeric_limits<double>::infinity());
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 1);

  // Test setters
  const double mu_dynamic = 10;
  solver_cast->set_mu_dynamic(mu_dynamic);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_dynamic(), 10);
  const double termination_tolerance = 1e-4;
  solver_cast->set_termination_tolerance(termination_tolerance);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-4);
  const bool use_filter_line_search = false;
  solver_cast->set_use_filter_line_search(use_filter_line_search);
  BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), false);
  const std::size_t filter_size = 2;
  solver_cast->set_filter_size(filter_size);
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 2);

  std::shared_ptr<mim_solvers::CallbackVerbose> callback_verbose =
      std::make_shared<mim_solvers::CallbackVerbose>(3);
  std::vector<std::shared_ptr<mim_solvers::CallbackAbstract>> callbacks;
  callbacks.push_back(callback_verbose);
  solver_cast->setCallbacks(callbacks);
}

//____________________________________________________________________________//

void test_csqp_core() {
  // Create solver
  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      factory.create(SolverTypes::SolverCSQP, ProblemTypes::ShootingProblem,
                     ModelTypes::PointMass2D, XConstraintType::AllIneq,
                     UConstraintType::AllEq);
  // Downcast
  std::shared_ptr<mim_solvers::SolverCSQP> solver_cast =
      std::static_pointer_cast<mim_solvers::SolverCSQP>(solver);

  // Test initial & default attributes
  std::vector<std::shared_ptr<crocoddyl::CallbackAbstract>> empty_callbacks;

  BOOST_CHECK_EQUAL(solver_cast->get_KKT(),
                    std::numeric_limits<double>::infinity());
  BOOST_CHECK_EQUAL(solver_cast->get_gap_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_constraint_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_qp_iters(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_xgrad_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_ugrad_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_merit(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), true);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_dynamic(), 1e1);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_constraint(), 1e1);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-6);
  BOOST_CHECK_EQUAL(solver_cast->get_max_qp_iters(), 1000);
  BOOST_CHECK_EQUAL(solver_cast->get_cost(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->get_equality_qp_initial_guess(), true);
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 1);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_update_interval(), 25);
  BOOST_CHECK_EQUAL(solver_cast->get_adaptive_rho_tolerance(), 5);
  BOOST_CHECK_EQUAL(solver_cast->get_alpha(), 1.6);
  BOOST_CHECK_EQUAL(solver_cast->get_sigma(), 1e-6);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_sparse(), 1e-1);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_abs(), 1e-4);
  BOOST_CHECK_EQUAL(solver_cast->get_norm_primal(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->get_norm_primal_tolerance(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->get_norm_dual(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->get_norm_dual_tolerance(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->get_reset_y(), false);
  BOOST_CHECK_EQUAL(solver_cast->get_reset_rho(), false);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_min(), 1e-6);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_max(), 1e3);

  // Test setters
  const double mu_dynamic = 100;
  solver_cast->set_mu_dynamic(mu_dynamic);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_dynamic(), 100);
  const double mu2 = 100;
  solver_cast->set_mu_constraint(mu2);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_constraint(), 100);
  const double alpha = 2.;
  solver_cast->set_alpha(alpha);
  BOOST_CHECK_EQUAL(solver_cast->get_alpha(), 2.);
  const double sigma = 1e-4;
  solver_cast->set_sigma(sigma);
  BOOST_CHECK_EQUAL(solver_cast->get_sigma(), 1e-4);
  const bool equality_qp_initial_guess = false;
  solver_cast->set_equality_qp_initial_guess(equality_qp_initial_guess);
  BOOST_CHECK_EQUAL(solver_cast->get_equality_qp_initial_guess(), false);
  const double termination_tolerance = 1e-4;
  solver_cast->set_termination_tolerance(termination_tolerance);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-4);
  const bool use_filter_line_search = false;
  solver_cast->set_use_filter_line_search(use_filter_line_search);
  BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), false);
  const std::size_t filter_size = 2;
  solver_cast->set_filter_size(filter_size);
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 2);
  const double rho_sparse = 1e-2;
  solver_cast->set_rho_sparse(rho_sparse);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_sparse(), 1e-2);
  const std::size_t rho_update_interval = 10;
  solver_cast->set_rho_update_interval(rho_update_interval);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_update_interval(), 10);
  const double adaptive_rho_tolerance = 10;
  solver_cast->set_adaptive_rho_tolerance(adaptive_rho_tolerance);
  BOOST_CHECK_EQUAL(solver_cast->get_adaptive_rho_tolerance(), 10);
  const std::size_t max_qp_iters = 100;
  solver_cast->set_max_qp_iters(max_qp_iters);
  BOOST_CHECK_EQUAL(solver_cast->get_max_qp_iters(), 100);
  const double eps_abs = 10;
  solver_cast->set_eps_abs(eps_abs);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_abs(), 10);
  const double eps_rel = 10;
  solver_cast->set_eps_rel(eps_rel);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_rel(), 10);
  std::shared_ptr<mim_solvers::CallbackVerbose> callback_verbose =
      std::make_shared<mim_solvers::CallbackVerbose>(3);
  std::vector<std::shared_ptr<mim_solvers::CallbackAbstract>> callbacks;
  callbacks.push_back(callback_verbose);
  solver_cast->setCallbacks(callbacks);
}

//____________________________________________________________________________//

#ifdef MIM_SOLVERS_WITH_PROXQP
void test_proxqp_core() {
  // Create solver
  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver =
      factory.create(SolverTypes::SolverPROXQP, ProblemTypes::ShootingProblem,
                     ModelTypes::PointMass2D, XConstraintType::AllIneq,
                     UConstraintType::AllEq);
  // Downcast
  std::shared_ptr<mim_solvers::SolverPROXQP> solver_cast =
      std::static_pointer_cast<mim_solvers::SolverPROXQP>(solver);

  // Test initial & default attributes
  BOOST_CHECK_EQUAL(solver_cast->get_KKT(),
                    std::numeric_limits<double>::infinity());
  BOOST_CHECK_EQUAL(solver_cast->get_gap_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_constraint_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_qp_iters(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_xgrad_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_ugrad_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_merit(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), true);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_dynamic(), 1e1);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_constraint(), 1e1);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-8);
  BOOST_CHECK_EQUAL(solver_cast->get_max_qp_iters(), 1000);
  BOOST_CHECK_EQUAL(solver_cast->get_cost(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 1);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_abs(), 1e-4);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_rel(), 1e-4);
  BOOST_CHECK_EQUAL(solver_cast->get_norm_primal(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->get_norm_dual(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->getCallbacks(), false);

  // Test setters
  const double mu_dynamic = 100;
  solver_cast->set_mu_dynamic(mu_dynamic);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_dynamic(), 100);
  const double mu_constraint = 100;
  solver_cast->set_mu_constraint(mu_constraint);
  BOOST_CHECK_EQUAL(solver_cast->get_mu_constraint(), 100);
  const double termination_tolerance = 1e-4;
  solver_cast->set_termination_tolerance(termination_tolerance);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-4);
  const bool use_filter_line_search = false;
  solver_cast->set_use_filter_line_search(use_filter_line_search);
  BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), false);
  const std::size_t filter_size = 2;
  solver_cast->set_filter_size(filter_size);
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 2);
  const std::size_t max_qp_iters = 100;
  solver_cast->set_max_qp_iters(max_qp_iters);
  BOOST_CHECK_EQUAL(solver_cast->get_max_qp_iters(), 100);
  const double eps_abs = 10;
  solver_cast->set_eps_abs(eps_abs);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_abs(), 10);
  const double eps_rel = 10;
  solver_cast->set_eps_rel(eps_rel);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_rel(), 10);
  const bool with_callbacks = true;
  solver_cast->setCallbacks(with_callbacks);
  BOOST_CHECK_EQUAL(solver_cast->getCallbacks(), true);
}
#endif

//____________________________________________________________________________//

void test_solver_convergence(SolverTypes::Type solver_type,
                             ProblemTypes::Type problem_type,
                             ModelTypes::Type model_type,
                             XConstraintType::Type x_cstr_type,
                             UConstraintType::Type u_cstr_type) {
  std::cout << "test_solver_convergence_" << solver_type << "_" << problem_type
            << "_" << model_type << "_" << x_cstr_type << "_" << u_cstr_type
            << std::endl;

  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solver = factory.create(
      solver_type, problem_type, model_type, x_cstr_type, u_cstr_type);

  // SQP params
  const int MAXITER = 10;
  const double SQP_TOL = 1e-4;
  // QP params (only for CSQP and PROXQP)
  const int QP_MAXITER = 1e5;
  const double EPS_ABS = 1e-10;
  const double EPS_REL = 0.;

  switch (solver_type) {
    case SolverTypes::SolverSQP: {
      std::shared_ptr<mim_solvers::SolverSQP> solver_cast =
          std::static_pointer_cast<mim_solvers::SolverSQP>(solver);
      solver_cast->set_termination_tolerance(SQP_TOL);

      std::shared_ptr<mim_solvers::CallbackVerbose> callback_verbose =
          std::make_shared<mim_solvers::CallbackVerbose>(3);
      std::vector<std::shared_ptr<mim_solvers::CallbackAbstract>> callbacks;
      callbacks.push_back(callback_verbose);
      solver_cast->setCallbacks(callbacks);

      solver_cast->solve(solver_cast->get_xs(), solver_cast->get_us(), MAXITER);
      BOOST_CHECK_EQUAL(solver->get_iter(), 1);
      BOOST_CHECK(solver_cast->get_KKT() <=
                  solver_cast->get_termination_tolerance());
      break;
    }
    case SolverTypes::SolverCSQP: {
      // if(x_cstr_type == XConstraintType::None && u_cstr_type ==
      // UConstraintType::AllEq){
      std::shared_ptr<mim_solvers::SolverCSQP> solver_cast =
          std::static_pointer_cast<mim_solvers::SolverCSQP>(solver);
      solver_cast->set_termination_tolerance(SQP_TOL);
      solver_cast->set_eps_rel(EPS_REL);
      solver_cast->set_eps_abs(EPS_ABS);
      solver_cast->set_max_qp_iters(QP_MAXITER);

      std::shared_ptr<mim_solvers::CallbackVerbose> callback_verbose =
          std::make_shared<mim_solvers::CallbackVerbose>(3);
      std::vector<std::shared_ptr<mim_solvers::CallbackAbstract>> callbacks;
      callbacks.push_back(callback_verbose);
      solver_cast->setCallbacks(callbacks);

      solver_cast->solve(solver_cast->get_xs(), solver_cast->get_us(), MAXITER);
      // Check SQP convergence
      BOOST_CHECK_EQUAL(solver->get_iter(), 1);
      BOOST_CHECK(solver_cast->get_KKT() <=
                  solver_cast->get_termination_tolerance());
      // Check QP convergence
      BOOST_CHECK(solver_cast->get_norm_primal() <=
                  solver_cast->get_norm_primal_tolerance());
      BOOST_CHECK(solver_cast->get_norm_dual() <=
                  solver_cast->get_norm_dual_tolerance());
      BOOST_CHECK(solver_cast->get_qp_iters() <
                  solver_cast->get_max_qp_iters());
      // }
      break;
    }
#ifdef MIM_SOLVERS_WITH_PROXQP
    case SolverTypes::SolverPROXQP: {
      std::shared_ptr<mim_solvers::SolverPROXQP> solver_cast =
          std::static_pointer_cast<mim_solvers::SolverPROXQP>(solver);
      solver_cast->set_termination_tolerance(SQP_TOL);
      solver_cast->set_eps_rel(EPS_REL);
      solver_cast->set_eps_abs(EPS_ABS);
      solver_cast->set_max_qp_iters(QP_MAXITER);
      solver_cast->setCallbacks(false);
      solver_cast->solve(solver_cast->get_xs(), solver_cast->get_us(), MAXITER);
      // Check SQP convergence
      BOOST_CHECK_EQUAL(solver->get_iter(), 1);
      BOOST_CHECK(solver_cast->get_KKT() <=
                  solver_cast->get_termination_tolerance());
      // Check QP convergence
      BOOST_CHECK(solver_cast->get_qp_iters() <
                  solver_cast->get_max_qp_iters());
      break;
    }
#endif
    default:
      std::cerr << "Error: Unknown solver type" << std::endl;
      break;
  }
}

//____________________________________________________________________________//

void test_csqp_equiv_sqp(ProblemTypes::Type problem_type,
                         ModelTypes::Type model_type) {
  std::cout << "test_csqp_equiv_sqp_" << problem_type << "_" << model_type
            << std::endl;
  SolverFactory factory;
  std::shared_ptr<crocoddyl::SolverAbstract> solverSQP =
      factory.create(SolverTypes::SolverSQP, problem_type, model_type,
                     XConstraintType::None, UConstraintType::None);
  std::shared_ptr<crocoddyl::SolverAbstract> solverCSQP =
      factory.create(SolverTypes::SolverCSQP, problem_type, model_type,
                     XConstraintType::None, UConstraintType::None);
  // Compare with constrained solver in the absence of constraints
  solverCSQP->solve();
  solverSQP->solve();
  for (std::size_t t = 0; t < solverCSQP->get_problem()->get_T(); t++) {
    BOOST_CHECK((solverCSQP->get_us()[t] - solverSQP->get_us()[t]).isZero(TOL));
    BOOST_CHECK((solverCSQP->get_xs()[t] - solverSQP->get_xs()[t]).isZero(TOL));
  }
}

//____________________________________________________________________________//

void register_sqp_core_test() {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_SQP_core";
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_sqp_core)));
  framework::master_test_suite().add(ts);
}

void register_csqp_core_test() {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_CSQP_core";
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_csqp_core)));
  framework::master_test_suite().add(ts);
}

#ifdef MIM_SOLVERS_WITH_PROXQP
void register_proxqp_core_test() {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_PROXQP_core";
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_proxqp_core)));
  framework::master_test_suite().add(ts);
}
#endif

void register_convergence_test(SolverTypes::Type solver_type,
                               ProblemTypes::Type problem_type,
                               ModelTypes::Type model_type,
                               XConstraintType::Type x_cstr_type,
                               UConstraintType::Type u_cstr_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << solver_type << "_" << problem_type << "_"
            << model_type << "_" << x_cstr_type << "_" << u_cstr_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_solver_convergence, solver_type,
                                      problem_type, model_type, x_cstr_type,
                                      u_cstr_type)));
  framework::master_test_suite().add(ts);
}

void register_equivalence_test(ProblemTypes::Type problem_type,
                               ModelTypes::Type model_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_equivalence_CSQP_vs_SQP_" << problem_type << "_"
            << model_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(
      boost::bind(&test_csqp_equiv_sqp, problem_type, model_type)));
  framework::master_test_suite().add(ts);
}

//____________________________________________________________________________//

bool init_function() {
  // Test solvers basic functionalities
  register_sqp_core_test();
  register_csqp_core_test();
#ifdef MIM_SOLVERS_WITH_PROXQP
  register_proxqp_core_test();
#endif

  // Test solvers convergence test
  for (size_t i_pb = 0; i_pb < ProblemTypes::all.size(); ++i_pb) {
    for (size_t i_md = 0; i_md < ModelTypes::all.size(); ++i_md) {
      // Unconstrained problems only (SQP)
      register_convergence_test(SolverTypes::SolverSQP, ProblemTypes::all[i_pb],
                                ModelTypes::all[i_md], XConstraintType::None,
                                UConstraintType::None);
      register_equivalence_test(ProblemTypes::all[i_pb], ModelTypes::all[i_md]);
      // Unconstrained AND Constrained problems (CSQP, PROXQP)
      for (size_t i_xc = 0; i_xc < XConstraintType::all.size(); ++i_xc) {
        for (size_t i_uc = 0; i_uc < UConstraintType::all.size(); ++i_uc) {
          register_convergence_test(
              SolverTypes::SolverCSQP, ProblemTypes::all[i_pb],
              ModelTypes::all[i_md], XConstraintType::all[i_xc],
              UConstraintType::all[i_uc]);
#ifdef MIM_SOLVERS_WITH_PROXQP
          register_convergence_test(
              SolverTypes::SolverPROXQP, ProblemTypes::all[i_pb],
              ModelTypes::all[i_md], XConstraintType::all[i_xc],
              UConstraintType::all[i_uc]);
#endif
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
