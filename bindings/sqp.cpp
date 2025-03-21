///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mim_solvers/sqp.hpp"

#include "mim_solvers/python.hpp"

namespace mim_solvers {

namespace bp = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverSQP_solves, SolverSQP::solve, 0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverSQP_computeDirections,
                                       SolverSQP::computeDirection, 0, 1)

void exposeSolverSQP() {
  bp::register_ptr_to_python<std::shared_ptr<SolverSQP> >();

  bp::class_<SolverSQP, bp::bases<SolverDDP> >(
      "SolverSQP",
      "SQP solver.\n\n"
      "The SQP solver computes an optimal trajectory and control commands by "
      "iterates\n"
      "running backward and forward passes. The backward-pass updates locally "
      "the\n"
      "quadratic approximation of the problem and computes descent direction,\n"
      "and the forward-pass rollouts this new policy by integrating the system "
      "dynamics\n"
      "along a tuple of optimized control commands U*.\n"
      ":param shootingProblem: shooting problem (list of action models along "
      "trajectory.)",
      bp::init<std::shared_ptr<crocoddyl::ShootingProblem> >(
          bp::args("self", "problem"),
          "Initialize the vector dimension.\n\n"
          ":param problem: shooting problem."))
      .def("solve", &SolverSQP::solve,
           SolverSQP_solves(
               bp::args("self", "init_xs", "init_us", "maxiter", "isFeasible",
                        "regInit"),
               "Compute the optimal trajectory xopt, uopt as lists of T+1 and "
               "T terms.\n\n"
               "From an initial guess init_xs,init_us (feasible or not), "
               "iterate\n"
               "over computeDirection and tryStep until stoppingCriteria is "
               "below\n"
               "threshold. It also describes the globalization strategy used\n"
               "during the numerical optimization.\n"
               ":param init_xs: initial guess for state trajectory with T+1 "
               "elements (default [])\n"
               ":param init_us: initial guess for control trajectory with T "
               "elements (default []).\n"
               ":param maxiter: maximum allowed number of iterations (default "
               "100).\n"
               ":param isFeasible: true if the init_xs are obtained from "
               "integrating the init_us (rollout) (default "
               "False).\n"
               ":param regInit: initial guess for the regularization value. "
               "Very low values are typical\n"
               "                used with very good guess points (init_xs, "
               "init_us) (default None).\n"
               ":returns the optimal trajectory xopt, uopt and a boolean that "
               "describes if convergence was reached."))
      .def_readwrite("xs_try", &SolverSQP::xs_try_, "xs try")
      .def_readwrite("us_try", &SolverSQP::us_try_, "us try")
      .def_readwrite("cost_try", &SolverSQP::cost_try_, "cost try")
      .def_readwrite("fs_try", &SolverSQP::fs_try_, "fs_try")
      .def_readwrite("lag_mul", &SolverSQP::lag_mul_, "lagrange multipliers")

      .add_property(
          "dx",
          make_function(&SolverSQP::get_dx,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "dx")
      .add_property(
          "du",
          make_function(&SolverSQP::get_du,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "du")

      .add_property("KKT", bp::make_function(&SolverSQP::get_KKT),
                    "KKT residual norm")
      .add_property("merit", bp::make_function(&SolverSQP::get_merit),
                    "Merit function value")

      // .add_property("with_callbacks",
      // bp::make_function(&SolverSQP::getCallbacks),
      // bp::make_function(&SolverSQP::setCallbacks), "Activates the callbacks
      // when true (default: False)")

      .add_property(
          "extra_iteration_for_last_kkt",
          bp::make_function(&SolverSQP::get_extra_iteration_for_last_kkt),
          bp::make_function(&SolverSQP::set_extra_iteration_for_last_kkt),
          "Additional iteration if SQP max. iter reached (default: False)")
      .add_property("mu_dynamic", bp::make_function(&SolverSQP::get_mu_dynamic),
                    bp::make_function(&SolverSQP::set_mu_dynamic),
                    "Penalty weight for dynamic violation in the merit "
                    "function (default: 1.)")
      .add_property("use_filter_line_search",
                    bp::make_function(&SolverSQP::get_use_filter_line_search),
                    bp::make_function(&SolverSQP::set_use_filter_line_search),
                    "Use the filter line search criteria (default: True)")
      .add_property(
          "termination_tolerance",
          bp::make_function(&SolverSQP::get_termination_tolerance),
          bp::make_function(&SolverSQP::set_termination_tolerance),
          "Termination criteria to exit the iteration (default: 1e-6)")
      .add_property("filter_size",
                    bp::make_function(&SolverSQP::get_filter_size),
                    bp::make_function(&SolverSQP::set_filter_size),
                    "filter size for the line-search (default: 1)");
}

}  // namespace mim_solvers
