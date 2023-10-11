///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <python/crocoddyl/core/core.hpp>
#include "mim_solvers/sqp.hpp"

namespace mim_solvers {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverSQP_solves, SolverSQP::solve, 0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverSQP_computeDirections, crocoddyl::SolverDDP::computeDirection, 0, 1)

void exposeSolverSQP() {
  bp::register_ptr_to_python<boost::shared_ptr<SolverSQP> >();

  bp::class_<SolverSQP, bp::bases<crocoddyl::SolverDDP> >(
      "SolverSQP",
      "SQP solver.\n\n"
      "The SQP solver computes an optimal trajectory and control commands by iterates\n"
      "running backward and forward passes. The backward-pass updates locally the\n"
      "quadratic approximation of the problem and computes descent direction,\n"
      "and the forward-pass rollouts this new policy by integrating the system dynamics\n"
      "along a tuple of optimized control commands U*.\n"
      ":param shootingProblem: shooting problem (list of action models along trajectory.)",
      bp::init<boost::shared_ptr<crocoddyl::ShootingProblem> >(bp::args("self", "problem"),
                                                    "Initialize the vector dimension.\n\n"
                                                    ":param problem: shooting problem."))
      .def("solve", &SolverSQP::solve,
           SolverSQP_solves(
               bp::args("self", "init_xs", "init_us", "maxiter", "isFeasible", "regInit"),
               "Compute the optimal trajectory xopt, uopt as lists of T+1 and T terms.\n\n"
               "From an initial guess init_xs,init_us (feasible or not), iterate\n"
               "over computeDirection and tryStep until stoppingCriteria is below\n"
               "threshold. It also describes the globalization strategy used\n"
               "during the numerical optimization.\n"
               ":param init_xs: initial guess for state trajectory with T+1 elements (default [])\n"
               ":param init_us: initial guess for control trajectory with T elements (default []).\n"
               ":param maxiter: maximum allowed number of iterations (default 100).\n"
               ":param isFeasible: true if the init_xs are obtained from integrating the init_us (rollout) (default "
               "False).\n"
               ":param regInit: initial guess for the regularization value. Very low values are typical\n"
               "                used with very good guess points (init_xs, init_us) (default None).\n"
               ":returns the optimal trajectory xopt, uopt and a boolean that describes if convergence was reached."))
     //  .def("updateExpectedImprovement", &SolverSQP::updateExpectedImprovement,
     //       bp::return_value_policy<bp::copy_const_reference>(), bp::args("self"),
     //       "Update the expected improvement model\n\n")
     // .def("increaseRegularization", &solverFDDP::increaseRegularization, bp::args("self"),
     //       "Increase regularization")
      .def_readwrite("xs_try", &SolverSQP::xs_try_, "xs try")
      .def_readwrite("us_try", &SolverSQP::us_try_, "us try")
      .def_readwrite("cost_try", &SolverSQP::cost_try_, "cost try")
      .def_readwrite("fs_try", &SolverSQP::fs_try_, "fs_try")
      .def_readwrite("lag_mul", &SolverSQP::lag_mul_, "lagrange multipliers")
      .def_readwrite("KKT", &SolverSQP::KKT_, "KKT residual")

      .add_property("with_callbacks", bp::make_function(&SolverSQP::getCallbacks), bp::make_function(&SolverSQP::setCallbacks),
                    "Activates the callbacks when true (default: False)")
      .add_property("use_kkt_criteria", bp::make_function(&SolverSQP::set_use_kkt_criteria), bp::make_function(&SolverSQP::get_use_kkt_criteria),
                    "Use the KKT residual condition as a termination criteria (default: True)")
      .add_property("mu", bp::make_function(&SolverSQP::get_mu), bp::make_function(&SolverSQP::set_mu),
                    "Penalty term for dynamic violation in the merit function (default: 1.)")
      .add_property("use_filter_line_search", bp::make_function(&SolverSQP::get_use_filter_line_search), bp::make_function(&SolverSQP::set_use_filter_line_search),
                    "Use the filter line search criteria (default: False)")
      .add_property("termination_tol", bp::make_function(&SolverSQP::get_termination_tolerance), bp::make_function(&SolverSQP::set_termination_tolerance),
                    "Termination criteria to exit the iteration (default: 1e-8)")
      .add_property("filter_size", bp::make_function(&SolverSQP::get_filter_size), bp::make_function(&SolverSQP::set_filter_size),
                    "filter size for the line-search (default: 10)");
     //  .add_property("th_acceptNegStep", bp::make_function(&SolverSQP::get_th_acceptnegstep),
     //                bp::make_function(&SolverSQP::set_th_acceptnegstep),
     //                "threshold for step acceptance in ascent direction");
     
}

}  // namespace python
}  // namespace mim_solvers