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

const double TOL = 1e-6;

//____________________________________________________________________________//

void test_sqp_core(){
  // Create solver
  SolverFactory factory;
  boost::shared_ptr<crocoddyl::SolverAbstract> solver = factory.create(SolverTypes::SolverSQP, 
                                                                       ProblemTypes::ShootingProblem, 
                                                                       ModelTypes::PointMass1D, 
                                                                       XConstraintType::None, 
                                                                       UConstraintType::None);
  // Downcast
  boost::shared_ptr<mim_solvers::SolverSQP> solver_cast = boost::static_pointer_cast<mim_solvers::SolverSQP>(solver); 
  
  // Test initial & default attributes
  BOOST_CHECK_EQUAL(solver_cast->get_KKT(), std::numeric_limits<double>::infinity());
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 1);
  BOOST_CHECK_EQUAL(solver_cast->get_gap_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_xgrad_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_ugrad_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_merit(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_mu(), 1e0);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-6);
  BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), true);
  BOOST_CHECK_EQUAL(solver_cast->get_use_kkt_criteria(), true);
  BOOST_CHECK_EQUAL(solver_cast->getCallbacks(), false);

  // Test setters
  const double mu = 10;
  solver_cast->set_mu(mu);
  BOOST_CHECK_EQUAL(solver_cast->get_mu(), 10);
  const double termination_tolerance = 1e-4;
  solver_cast->set_termination_tolerance(termination_tolerance);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-4);
  const bool use_kkt_criteria = false;
  solver_cast->set_use_kkt_criteria(use_kkt_criteria);
  BOOST_CHECK_EQUAL(solver_cast->get_use_kkt_criteria(), false);
  const bool use_filter_line_search = false;
  solver_cast->set_use_filter_line_search(use_filter_line_search);
  BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), false);
  const std::size_t filter_size = 2;
  solver_cast->set_filter_size(filter_size);
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 2);
  const bool with_callbacks = true;
  solver_cast->setCallbacks(with_callbacks);
  BOOST_CHECK_EQUAL(solver_cast->getCallbacks(), true);
}

//____________________________________________________________________________//

void test_csqp_core(){
  // Create solver
  SolverFactory factory;
  boost::shared_ptr<crocoddyl::SolverAbstract> solver = factory.create(SolverTypes::SolverCSQP, 
                                                                       ProblemTypes::ShootingProblem, 
                                                                       ModelTypes::PointMass1D, 
                                                                       XConstraintType::AllIneq, 
                                                                       UConstraintType::AllEq);
  // Downcast
  boost::shared_ptr<mim_solvers::SolverCSQP> solver_cast = boost::static_pointer_cast<mim_solvers::SolverCSQP>(solver); 
  
  // Test initial & default attributes
  BOOST_CHECK_EQUAL(solver_cast->get_KKT(), std::numeric_limits<double>::infinity());
  BOOST_CHECK_EQUAL(solver_cast->get_gap_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_constraint_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_qp_iters(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_xgrad_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_ugrad_norm(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_merit(), 0);
  BOOST_CHECK_EQUAL(solver_cast->get_use_kkt_criteria(), true);
  BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), true);
  BOOST_CHECK_EQUAL(solver_cast->get_mu(), 1e1);
  BOOST_CHECK_EQUAL(solver_cast->get_mu2(), 1e1);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-6);
  BOOST_CHECK_EQUAL(solver_cast->get_max_qp_iters(), 1000);
  BOOST_CHECK_EQUAL(solver_cast->get_cost(), 0.);
  BOOST_CHECK_EQUAL(solver_cast->get_warm_start(), true);
  BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 1);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_update_interval(), 25);
  BOOST_CHECK_EQUAL(solver_cast->get_adaptive_rho_tolerance(), 5);
  BOOST_CHECK_EQUAL(solver_cast->get_alpha(), 1.6);
  BOOST_CHECK_EQUAL(solver_cast->get_sigma(), 1e-6);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_sparse(), 1e-1);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_abs(), 1e-4);
  BOOST_CHECK_EQUAL(solver_cast->get_eps_rel(), 1e-4);
  BOOST_CHECK_EQUAL(solver_cast->get_warm_start_y(), false);
  BOOST_CHECK_EQUAL(solver_cast->get_reset_rho(), false);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_min(), 1e-6);
  BOOST_CHECK_EQUAL(solver_cast->get_rho_max(), 1e3);
  BOOST_CHECK_EQUAL(solver_cast->getCallbacks(), false);

  // Test setters
  const double mu = 10;
  solver_cast->set_mu(mu);
  BOOST_CHECK_EQUAL(solver_cast->get_mu(), 10);
  const double mu2 = 10;
  solver_cast->set_mu2(mu2);
  BOOST_CHECK_EQUAL(solver_cast->get_mu2(), 10);
  const double alpha = 2.;
  solver_cast->set_alpha(alpha);
  BOOST_CHECK_EQUAL(solver_cast->get_alpha(), 2.);
  const double sigma = 1e-4;
  solver_cast->set_sigma(sigma);
  BOOST_CHECK_EQUAL(solver_cast->get_sigma(), 1e-4);
  const bool warm_start = false;
  solver_cast->set_warm_start(warm_start);
  BOOST_CHECK_EQUAL(solver_cast->get_warm_start(), false);
  const double termination_tolerance = 1e-4;
  solver_cast->set_termination_tolerance(termination_tolerance);
  BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-4);
  const bool use_kkt_criteria = false;
  solver_cast->set_use_kkt_criteria(use_kkt_criteria);
  BOOST_CHECK_EQUAL(solver_cast->get_use_kkt_criteria(), false);
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
  const bool with_callbacks = true;
  solver_cast->setCallbacks(with_callbacks);
  BOOST_CHECK_EQUAL(solver_cast->getCallbacks(), true);
}

//____________________________________________________________________________//

#ifdef MIM_SOLVERS_WITH_PROXQP
  void test_proxqp_core(){
    // Create solver
    SolverFactory factory;
    boost::shared_ptr<crocoddyl::SolverAbstract> solver = factory.create(SolverTypes::SolverPROXQP, 
                                                                        ProblemTypes::ShootingProblem, 
                                                                        ModelTypes::PointMass1D, 
                                                                        XConstraintType::AllIneq, 
                                                                        UConstraintType::AllEq);
    // Downcast
    boost::shared_ptr<mim_solvers::SolverPROXQP> solver_cast = boost::static_pointer_cast<mim_solvers::SolverPROXQP>(solver); 
    
    // Test initial & default attributes
    BOOST_CHECK_EQUAL(solver_cast->get_KKT(), std::numeric_limits<double>::infinity());
    BOOST_CHECK_EQUAL(solver_cast->get_gap_norm(), 0);
    BOOST_CHECK_EQUAL(solver_cast->get_constraint_norm(), 0);
    BOOST_CHECK_EQUAL(solver_cast->get_qp_iters(), 0);
    BOOST_CHECK_EQUAL(solver_cast->get_xgrad_norm(), 0);
    BOOST_CHECK_EQUAL(solver_cast->get_ugrad_norm(), 0);
    BOOST_CHECK_EQUAL(solver_cast->get_merit(), 0);
    BOOST_CHECK_EQUAL(solver_cast->get_use_kkt_criteria(), true);
    BOOST_CHECK_EQUAL(solver_cast->get_use_filter_line_search(), true);
    BOOST_CHECK_EQUAL(solver_cast->get_mu(), 1e1);
    BOOST_CHECK_EQUAL(solver_cast->get_mu2(), 1e1);
    BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-8);
    BOOST_CHECK_EQUAL(solver_cast->get_max_qp_iters(), 1000);
    BOOST_CHECK_EQUAL(solver_cast->get_cost(), 0.);
    BOOST_CHECK_EQUAL(solver_cast->get_filter_size(), 1);
    BOOST_CHECK_EQUAL(solver_cast->get_eps_abs(), 1e-4);
    BOOST_CHECK_EQUAL(solver_cast->getCallbacks(), false);

    // Test setters
    const double mu = 10;
    solver_cast->set_mu(mu);
    BOOST_CHECK_EQUAL(solver_cast->get_mu(), 10);
    const double mu2 = 10;
    solver_cast->set_mu2(mu2);
    BOOST_CHECK_EQUAL(solver_cast->get_mu2(), 10);
    const double termination_tolerance = 1e-4;
    solver_cast->set_termination_tolerance(termination_tolerance);
    BOOST_CHECK_EQUAL(solver_cast->get_termination_tolerance(), 1e-4);
    const bool use_kkt_criteria = false;
    solver_cast->set_use_kkt_criteria(use_kkt_criteria);
    BOOST_CHECK_EQUAL(solver_cast->get_use_kkt_criteria(), false);
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
  
  SolverFactory factory;
  boost::shared_ptr<crocoddyl::SolverAbstract> solver = factory.create(solver_type, problem_type, model_type, x_cstr_type, u_cstr_type);
  
  // solver->setCallbacks(true);
  // solver->solve();

  // BOOST_CHECK_EQUAL(solver->get_iter(), 1);

  // need to cast to get residual and tolerance
  switch (solver_type)
  {
  case SolverTypes::SolverSQP:
  {
    boost::shared_ptr<mim_solvers::SolverSQP> solver_cast = boost::static_pointer_cast<mim_solvers::SolverSQP>(solver); 
    solver_cast->set_termination_tolerance(1e-4);
    solver_cast->setCallbacks(true);
    solver_cast->solve();
    BOOST_CHECK_EQUAL(solver->get_iter(), 1);
    BOOST_CHECK(solver_cast->get_KKT() <= solver_cast->get_termination_tolerance());
    break;
  }
  case SolverTypes::SolverCSQP:
  {
    boost::shared_ptr<mim_solvers::SolverCSQP> solver_cast = boost::static_pointer_cast<mim_solvers::SolverCSQP>(solver); 
    solver_cast->set_termination_tolerance(1e-4);
    solver_cast->set_eps_rel(0.);
    solver_cast->set_eps_abs(1e-6);
    solver_cast->set_max_qp_iters(1e4);
    solver_cast->setCallbacks(true);
    solver_cast->solve(solver_cast->get_xs(), solver_cast->get_us(), 2);
    BOOST_CHECK_EQUAL(solver->get_iter(), 1);
    BOOST_CHECK(solver_cast->get_KKT() <= solver_cast->get_termination_tolerance());
    break;
  }
#ifdef MIM_SOLVERS_WITH_PROXQP
  case SolverTypes::SolverPROXQP:
  {
    boost::shared_ptr<mim_solvers::SolverPROXQP> solver_cast = boost::static_pointer_cast<mim_solvers::SolverPROXQP>(solver); 
    BOOST_CHECK(solver_cast->get_KKT() <= solver_cast->get_termination_tolerance());
    break;
  }
#endif
  default:
    std::cerr << "Error: Unknown solver type" << std::endl; 
    break;
  }
}

// //____________________________________________________________________________//

// void test_csqp_equiv_sqp(ProblemTypes::Type problem_type,
//                          ModelTypes::Type model_type,
//                          XConstraintType::Type x_cstr_type,
//                          UConstraintType::Type u_cstr_type) {
  
//   const double TOL = 1e-6;
//   SolverFactory factory;
//   boost::shared_ptr<crocoddyl::SolverAbstract> solverCSQP = factory.create(solver_type, problem_type, model_type, x_cstr_type, u_cstr_type);
  
//   // Compare with constrained solver in the absence of constraints
//   if(x_cstr_type == XConstraintType::None && u_cstr_type == UConstraintType::None){
//     boost::shared_ptr<crocoddyl::SolverAbstract> solverSQP = factory.create(SolverTypes::SolverSQP, problem_type, model_type, x_cstr_type, u_cstr_type);
//     solverCSQP->solve();
//     solverSQP->solve();
//     for(std::size_t t=0; t<solverCSQP->get_problem()->get_T(); t++){
//       BOOST_CHECK((solverCSQP->get_us()[t] - solverSQP->get_us()[t]).isZero(TOL));
//       BOOST_CHECK((solverCSQP->get_xs()[t] - solverSQP->get_xs()[t]).isZero(TOL));
//     }
//     // Convergence in 1 iteration
//     BOOST_CHECK_EQUAL(solverSQP->get_iter(), 1);
//     // KKT residual is below termination tolerance
//     boost::shared_ptr<mim_solvers::SolverSQP> solverSQP_cast = boost::static_pointer_cast<mim_solvers::SolverSQP>(solverSQP); 
//     BOOST_CHECK(solverSQP_cast->get_KKT() <= solverSQP_cast->get_termination_tolerance());
//   }
  
//   boost::shared_ptr<mim_solvers::SolverCSQP> solverCSQP_cast = boost::static_pointer_cast<mim_solvers::SolverCSQP>(solverCSQP); 
//   BOOST_CHECK_EQUAL(solverCSQP->get_iter(), 1);
//   BOOST_CHECK(solverCSQP_cast->get_KKT() <= solverCSQP_cast->get_termination_tolerance());
// }

// //____________________________________________________________________________//

// #ifdef MIM_SOLVERS_WITH_PROXQP
//   void test_proxqp_equiv_sqp(ProblemTypes::Type problem_type,
//                             ModelTypes::Type model_type,
//                             XConstraintType::Type x_cstr_type,
//                             UConstraintType::Type u_cstr_type) {
    
    
//     SolverFactory factory;
//     boost::shared_ptr<crocoddyl::SolverAbstract> solverCSQP = factory.create(SolverTypes::SolverPROXQP, problem_type, model_type, x_cstr_type, u_cstr_type);
    
//     // Compare with constrained solver in the absence of constraints
//     if(x_cstr_type == XConstraintType::None && u_cstr_type == UConstraintType::None){
//       boost::shared_ptr<crocoddyl::SolverAbstract> solverSQP = factory.create(SolverTypes::SolverSQP, problem_type, model_type, x_cstr_type, u_cstr_type);
//       solverCSQP->solve();
//       solverSQP->solve();
//       for(std::size_t t=0; t<solverCSQP->get_problem()->get_T(); t++){
//         BOOST_CHECK((solverCSQP->get_us()[t] - solverSQP->get_us()[t]).isZero(TOL));
//         BOOST_CHECK((solverCSQP->get_xs()[t] - solverSQP->get_xs()[t]).isZero(TOL));
//       }
//       // Convergence in 1 iteration
//       BOOST_CHECK_EQUAL(solverSQP->get_iter(), 1);
//       // KKT residual is below termination tolerance
//       boost::shared_ptr<mim_solvers::SolverSQP> solverSQP_cast = boost::static_pointer_cast<mim_solvers::SolverSQP>(solverSQP); 
//       BOOST_CHECK(solverSQP_cast->get_KKT() <= solverSQP_cast->get_termination_tolerance());
//     }
    
//     boost::shared_ptr<mim_solvers::SolverCSQP> solverCSQP_cast = boost::static_pointer_cast<mim_solvers::SolverCSQP>(solverCSQP); 
//     BOOST_CHECK_EQUAL(solverCSQP->get_iter(), 1);
//     BOOST_CHECK(solverCSQP_cast->get_KKT() <= solverCSQP_cast->get_termination_tolerance());
//   }


//   void test_csqp_equiv_proxqp(ProblemTypes::Type problem_type,
//                               ModelTypes::Type model_type,
//                               XConstraintType::Type x_cstr_type,
//                               UConstraintType::Type u_cstr_type) {
    
//     const double TOL = 1e-6;
//     SolverFactory factory;
//     boost::shared_ptr<crocoddyl::SolverAbstract> solverCSQP = factory.create(solver_type, problem_type, model_type, x_cstr_type, u_cstr_type);
    
//     // Compare with constrained solver in the absence of constraints
//     if(x_cstr_type == XConstraintType::None && u_cstr_type == UConstraintType::None){
//       boost::shared_ptr<crocoddyl::SolverAbstract> solverSQP = factory.create(SolverTypes::SolverSQP, problem_type, model_type, x_cstr_type, u_cstr_type);
//       solverCSQP->solve();
//       solverSQP->solve();
//       for(std::size_t t=0; t<solverCSQP->get_problem()->get_T(); t++){
//         BOOST_CHECK((solverCSQP->get_us()[t] - solverSQP->get_us()[t]).isZero(TOL));
//         BOOST_CHECK((solverCSQP->get_xs()[t] - solverSQP->get_xs()[t]).isZero(TOL));
//       }
//       // Convergence in 1 iteration
//       BOOST_CHECK_EQUAL(solverSQP->get_iter(), 1);
//       // KKT residual is below termination tolerance
//       boost::shared_ptr<mim_solvers::SolverSQP> solverSQP_cast = boost::static_pointer_cast<mim_solvers::SolverSQP>(solverSQP); 
//       BOOST_CHECK(solverSQP_cast->get_KKT() <= solverSQP_cast->get_termination_tolerance());
//     }
    
//     boost::shared_ptr<mim_solvers::SolverCSQP> solverCSQP_cast = boost::static_pointer_cast<mim_solvers::SolverCSQP>(solverCSQP); 
//     BOOST_CHECK_EQUAL(solverCSQP->get_iter(), 1);
//     BOOST_CHECK(solverCSQP_cast->get_KKT() <= solverCSQP_cast->get_termination_tolerance());
//   }

// #endif

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

//____________________________________________________________________________//


void register_convergence_test(SolverTypes::Type solver_type,
                               ProblemTypes::Type problem_type,
                               ModelTypes::Type model_type,
                               XConstraintType::Type x_cstr_type,
                               UConstraintType::Type u_cstr_type) {
  boost::test_tools::output_test_stream test_name;
  test_name << "test_" << solver_type << "_" << problem_type << "_" << model_type << "_" << x_cstr_type << "_" << u_cstr_type;
  test_suite* ts = BOOST_TEST_SUITE(test_name.str());
  std::cout << "Running " << test_name.str() << std::endl;
  ts->add(BOOST_TEST_CASE(boost::bind(&test_solver_convergence, solver_type, problem_type, model_type, x_cstr_type, u_cstr_type)));
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
      if(ModelTypes::all[i_md] != ModelTypes::PointMass1D){
        // Unconstrained problems only (SQP)
        register_convergence_test(SolverTypes::SolverSQP,
                                      ProblemTypes::all[i_pb],
                                      ModelTypes::all[i_md],
                                      XConstraintType::None,
                                      UConstraintType::None);
        // Unconstrained AND Constrained problems (CSQP, PROXQP)
        for (size_t i_xc = 0; i_xc < XConstraintType::all.size(); ++i_xc) {
          for (size_t i_uc = 0; i_uc < UConstraintType::all.size(); ++i_uc) {
              register_convergence_test(SolverTypes::SolverCSQP,
                                            ProblemTypes::all[i_pb],
                                            ModelTypes::all[i_md],
                                            XConstraintType::all[i_xc],
                                            UConstraintType::all[i_uc]);
              register_convergence_test(SolverTypes::SolverPROXQP,
                                            ProblemTypes::all[i_pb],
                                            ModelTypes::all[i_md],
                                            XConstraintType::all[i_xc],
                                            UConstraintType::all[i_uc]);
          }
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
