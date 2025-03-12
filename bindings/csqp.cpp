///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mim_solvers/csqp.hpp"

#include "mim_solvers/python.hpp"

namespace mim_solvers {

namespace bp = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverCSQP_solves, SolverCSQP::solve, 0,
                                       5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverCSQP_computeDirections,
                                       SolverCSQP::computeDirection, 0, 1)

void exposeSolverCSQP() {
  bp::register_ptr_to_python<std::shared_ptr<SolverCSQP> >();

  bp::class_<SolverCSQP, bp::bases<SolverDDP> >(
      "SolverCSQP",
      "CSQP solver.\n\n"
      "The CSQP solver computes an optimal trajectory and control commands by "
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
      .def("solve", &SolverCSQP::solve,
           SolverCSQP_solves(
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

      .def("calc", &SolverCSQP::calc, bp::args("self", "recalc"), "")
      .def("update_lagrangian_parameters",
           &SolverCSQP::update_lagrangian_parameters, bp::args("self"), "")
      .def("forwardPass", &SolverCSQP::forwardPass, bp::args("self"), "")
      .def("backwardPass", &SolverCSQP::backwardPass, bp::args("self"), "")
      .def("backwardPass_without_constraints",
           &SolverCSQP::backwardPass_without_constraints, bp::args("self"), "")
      .def("backwardPass_without_rho_update",
           &SolverCSQP::backwardPass_without_rho_update, bp::args("self"), "")
      .def("update_rho_vec", &SolverCSQP::update_rho_vec,
           bp::args("self", "iter"), "")
      .def("computeDirection", &SolverCSQP::computeDirection,
           bp::args("self", "recalcDiff"), "")
      .def("checkKKTConditions", &SolverCSQP::checkKKTConditions,
           bp::args("self"), "")
      .def_readwrite("xs_try", &SolverCSQP::xs_try_, "xs try")
      .def_readwrite("us_try", &SolverCSQP::us_try_, "us try")
      .def_readwrite("cost_try", &SolverCSQP::cost_try_, "cost try")
      .def_readwrite("fs_try", &SolverCSQP::fs_try_, "fs_try")
      .def_readwrite("lag_mul", &SolverCSQP::lag_mul_, "lagrange multipliers")
      .def_readwrite("norm_primal", &SolverCSQP::norm_primal_, "norm_primal")
      .def_readwrite("norm_dual", &SolverCSQP::norm_dual_, "norm_dual ")
      .def_readwrite("norm_dual_rel", &SolverCSQP::norm_dual_rel_,
                     "norm_dual_rel")
      .def_readwrite("norm_primal_rel", &SolverCSQP::norm_primal_rel_,
                     "norm_primal_rel")
      .def_readwrite("rho_vec", &SolverCSQP::rho_vec_, "rho vector")
      .def_readwrite("y", &SolverCSQP::y_, "y")
      .def_readwrite("z", &SolverCSQP::z_, "z")
      .def_readwrite("reset_y", &SolverCSQP::reset_y_,
                     "Reset ADMM Lagrange multipliers to zero (default: False, "
                     "i.e. reset to previous)")
      .def_readwrite(
          "reset_rho", &SolverCSQP::reset_rho_,
          "Reset the rho parameter (default: False, i.e. reset to previous)")
      .def_readwrite("update_rho_with_heuristic",
                     &SolverCSQP::update_rho_with_heuristic_,
                     "Update the heuristic for the rho update (default: False)")
      .def_readwrite("remove_reg", &SolverCSQP::remove_reg_,
                     "Removes Crocoddyl's regularization by setting "
                     "(preg,dreg)=0 when True (default: False)")

      //  .add_property("with_callbacks",
      //  bp::make_function(&SolverCSQP::getCallbacks),
      //  bp::make_function(&SolverCSQP::setCallbacks),
      //                "Activates the callbacks when true (default: False)")

      .add_property("with_qp_callbacks",
                    bp::make_function(&SolverCSQP::getQPCallbacks),
                    bp::make_function(&SolverCSQP::setQPCallbacks),
                    "Activates the QP callbacks when true (default: False)")
      .add_property(
          "extra_iteration_for_last_kkt",
          bp::make_function(&SolverCSQP::get_extra_iteration_for_last_kkt),
          bp::make_function(&SolverCSQP::set_extra_iteration_for_last_kkt),
          "Additional iteration if SQP max. iter reached (default: False)")
      .add_property(
          "xs",
          make_function(&SolverCSQP::get_xs,
                        bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverCSQP::set_xs), "xs")
      .add_property(
          "us",
          make_function(&SolverCSQP::get_us,
                        bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverCSQP::set_us), "us")
      .add_property(
          "dx_tilde",
          make_function(&SolverCSQP::get_dx_tilde,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "dx_tilde")
      .add_property(
          "du_tilde",
          make_function(&SolverCSQP::get_du_tilde,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "du_tilde")
      .add_property(
          "dx",
          make_function(&SolverCSQP::get_dx,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "dx")
      .add_property(
          "du",
          make_function(&SolverCSQP::get_du,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "du")

      .add_property("constraint_norm",
                    bp::make_function(&SolverCSQP::get_constraint_norm),
                    "Constraint norm")
      .add_property("gap_norm", bp::make_function(&SolverCSQP::get_gap_norm),
                    "Gap norm")
      .add_property("qp_iters", bp::make_function(&SolverCSQP::get_qp_iters),
                    "Number of QP iterations")
      .add_property("KKT", bp::make_function(&SolverCSQP::get_KKT),
                    "KKT residual norm")
      .add_property("merit", bp::make_function(&SolverCSQP::get_merit),
                    "Merit function value")

      .add_property("lag_mul_inf_norm_coef",
                    bp::make_function(&SolverCSQP::get_lag_mul_inf_norm_coef),
                    bp::make_function(&SolverCSQP::set_lag_mul_inf_norm_coef),
                    "Scaling coefficient for the Lagrange multipliers norm in "
                    "Nocedal's L1 merit function (default: 10.)")
      .add_property("mu_dynamic",
                    bp::make_function(&SolverCSQP::get_mu_dynamic),
                    bp::make_function(&SolverCSQP::set_mu_dynamic),
                    "Penalty weight for dynamic violation in the merit "
                    "function (default: 10.)")
      .add_property("mu_constraint",
                    bp::make_function(&SolverCSQP::get_mu_constraint),
                    bp::make_function(&SolverCSQP::set_mu_constraint),
                    "Penalty weight for constraint violation in the merit "
                    "function (default: 10.)")

      .add_property("eps_abs", bp::make_function(&SolverCSQP::get_eps_abs),
                    bp::make_function(&SolverCSQP::set_eps_abs),
                    "sets epsillon absolute termination criteria for qp solver")
      .add_property("eps_rel", bp::make_function(&SolverCSQP::get_eps_rel),
                    bp::make_function(&SolverCSQP::set_eps_rel),
                    "sets epsillon relative termination criteria for qp solver")
      .add_property("rho_sparse",
                    bp::make_function(&SolverCSQP::get_rho_sparse),
                    bp::make_function(&SolverCSQP::set_rho_sparse),
                    "Penalty term for dynamic violation in the merit function "
                    "(default: 1.)")
      .add_property(
          "equality_qp_initial_guess",
          bp::make_function(&SolverCSQP::get_equality_qp_initial_guess),
          bp::make_function(&SolverCSQP::set_equality_qp_initial_guess),
          "initialize each qp with the solution of the equality qp. (default: "
          "True)")
      .add_property("sigma", bp::make_function(&SolverCSQP::get_sigma),
                    bp::make_function(&SolverCSQP::set_sigma),
                    "get and set sigma")
      .add_property("alpha", bp::make_function(&SolverCSQP::get_alpha),
                    bp::make_function(&SolverCSQP::set_alpha),
                    "get and set alpha (relaxed update)")

      .add_property("use_filter_line_search",
                    bp::make_function(&SolverCSQP::get_use_filter_line_search),
                    bp::make_function(&SolverCSQP::set_use_filter_line_search),
                    "Use the filter line search criteria (default: True)")
      .add_property(
          "termination_tolerance",
          bp::make_function(&SolverCSQP::get_termination_tolerance),
          bp::make_function(&SolverCSQP::set_termination_tolerance),
          "Termination criteria to exit the iteration (default: 1e-6)")
      .add_property("max_qp_iters",
                    bp::make_function(&SolverCSQP::get_max_qp_iters),
                    bp::make_function(&SolverCSQP::set_max_qp_iters),
                    "get and set max qp iters")
      .add_property("rho_update_interval",
                    bp::make_function(&SolverCSQP::get_rho_update_interval),
                    bp::make_function(&SolverCSQP::set_rho_update_interval),
                    "get and set rho update interval")
      .add_property("filter_size",
                    bp::make_function(&SolverCSQP::get_filter_size),
                    bp::make_function(&SolverCSQP::set_filter_size),
                    "filter size for the line-search (default: 1)")
      .add_property("adaptive_rho_tolerance",
                    bp::make_function(&SolverCSQP::get_adaptive_rho_tolerance),
                    bp::make_function(&SolverCSQP::set_adaptive_rho_tolerance),
                    "get and set adaptive rho tolerance")
      .add_property("max_solve_time",
                    bp::make_function(&SolverCSQP::get_max_solve_time),
                    bp::make_function(&SolverCSQP::set_max_solve_time),
                    "get and set max solve time in seconds")
      .add_property("max_solve_time_reached",
                    bp::make_function(&SolverCSQP::get_max_solve_time_reached),
                    "get if solver timed out");
}

}  // namespace mim_solvers
