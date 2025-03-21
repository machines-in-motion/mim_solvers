///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mim_solvers/csqp_proxqp.hpp"

#include "mim_solvers/python.hpp"

namespace mim_solvers {

namespace bp = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverPROXQP_solves, SolverPROXQP::solve,
                                       0, 5)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SolverPROXQP_computeDirections,
                                       SolverDDP::computeDirection, 0, 1)

void exposeSolverPROXQP() {
  bp::register_ptr_to_python<std::shared_ptr<SolverPROXQP> >();

  bp::class_<SolverPROXQP, bp::bases<SolverDDP> >(
      "SolverPROXQP",
      "SolverPROXQP solver.\n\n"
      "The SolverPROXQP solver computes an optimal trajectory and control "
      "commands \n"
      "by using ProxQP to solve the full size sparse sub QP problem at each "
      "SQP iteration. U*.\n"
      ":param shootingProblem: shooting problem (list of action models along "
      "trajectory.)",
      bp::init<std::shared_ptr<crocoddyl::ShootingProblem> >(
          bp::args("self", "problem"),
          "Initialize the vector dimension.\n\n"
          ":param problem: shooting problem."))
      .def("solve", &SolverPROXQP::solve,
           SolverPROXQP_solves(
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

               ":param regInit: initial guess for the regularization value. "
               "Very low values are typical\n"
               "                used with very good guess points (init_xs, "
               "init_us) (default None).\n"
               ":returns the optimal trajectory xopt, uopt and a boolean that "
               "describes if convergence was reached."))
      //  .def("updateExpectedImprovement",
      //  &SolverPROXQP::updateExpectedImprovement,
      //       bp::return_value_policy<bp::copy_const_reference>(),
      //       bp::args("self"), "Update the expected improvement model\n\n")
      // .def("increaseRegularization", &solverFDDP::increaseRegularization,
      // bp::args("self"),
      //       "Increase regularization")

      .def("calc", &SolverPROXQP::calc, bp::args("self", "recalc"), "")
      .def("computeDirection", &SolverPROXQP::computeDirection,
           bp::args("self", "recalcDiff"), "")

      .def_readwrite("xs_try", &SolverPROXQP::xs_try_, "xs try")
      .def_readwrite("us_try", &SolverPROXQP::us_try_, "us try")
      .def_readwrite("cost_try", &SolverPROXQP::cost_try_, "cost try")
      .def_readwrite("fs_try", &SolverPROXQP::fs_try_, "fs_try")
      .def_readwrite("lag_mul", &SolverPROXQP::lag_mul_, "lagrange multipliers")

      .add_property("constraint_norm",
                    bp::make_function(&SolverPROXQP::get_constraint_norm),
                    "Constraint norm")
      .add_property("gap_norm", bp::make_function(&SolverPROXQP::get_gap_norm),
                    "Gap norm")
      .add_property("qp_iters", bp::make_function(&SolverPROXQP::get_qp_iters),
                    "Number of QP iterations")
      .add_property("KKT_norm", bp::make_function(&SolverPROXQP::get_KKT),
                    "KKT norm")

      .add_property("with_callbacks",
                    bp::make_function(&SolverPROXQP::getCallbacks),
                    bp::make_function(&SolverPROXQP::setCallbacks),
                    "Activates the callbacks when true (default: False)")
      .add_property(
          "xs",
          make_function(&SolverPROXQP::get_xs,
                        bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverPROXQP::set_xs), "xs")
      .add_property(
          "us",
          make_function(&SolverPROXQP::get_us,
                        bp::return_value_policy<bp::copy_const_reference>()),
          bp::make_function(&SolverPROXQP::set_us), "us")
      .add_property(
          "P",
          make_function(&SolverPROXQP::get_P,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "P")
      .add_property(
          "A",
          make_function(&SolverPROXQP::get_A,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "A")
      .add_property(
          "C",
          make_function(&SolverPROXQP::get_C,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "C")
      .add_property(
          "q",
          make_function(&SolverPROXQP::get_q,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "q")
      .add_property(
          "b",
          make_function(&SolverPROXQP::get_b,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "b")
      .add_property(
          "l",
          make_function(&SolverPROXQP::get_l,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "l")
      .add_property(
          "u",
          make_function(&SolverPROXQP::get_u,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "u")

      .add_property(
          "dx",
          make_function(&SolverPROXQP::get_dx,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "dx")
      .add_property(
          "du",
          make_function(&SolverPROXQP::get_du,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "du")
      .add_property(
          "y",
          make_function(&SolverPROXQP::get_y,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "y")
      .add_property(
          "lag_mul",
          make_function(&SolverPROXQP::get_lag_mul,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "lag_mul")
      .add_property(
          "fs",
          make_function(&SolverPROXQP::get_fs,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "fs")

      .add_property("eps_abs", bp::make_function(&SolverPROXQP::get_eps_abs),
                    bp::make_function(&SolverPROXQP::set_eps_abs),
                    "sets epsillon absolute termination criteria for qp solver")

      .add_property(
          "use_filter_line_search",
          bp::make_function(&SolverPROXQP::get_use_filter_line_search),
          bp::make_function(&SolverPROXQP::set_use_filter_line_search),
          "Use the filter line search criteria (default: True)")
      .add_property(
          "termination_tolerance",
          bp::make_function(&SolverPROXQP::get_termination_tolerance),
          bp::make_function(&SolverPROXQP::set_termination_tolerance),
          "Termination criteria to exit the iteration (default: 1e-8)")
      .add_property("max_qp_iters",
                    bp::make_function(&SolverPROXQP::get_max_qp_iters),
                    bp::make_function(&SolverPROXQP::set_max_qp_iters),
                    "get and set max qp iters");
}

}  // namespace mim_solvers
