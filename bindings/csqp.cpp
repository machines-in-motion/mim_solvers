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

// Raw function wrapper for setCallbacks to properly override base class method
static bp::object setCallbacks_wrapper(bp::tuple args, bp::dict kwargs) {
  if (bp::len(args) != 2) {
    PyErr_SetString(PyExc_TypeError, "setCallbacks() takes exactly 2 arguments");
    bp::throw_error_already_set();
  }
  
  SolverCSQP& self = bp::extract<SolverCSQP&>(args[0]);
  bp::object callbacks_list = args[1];
  
  std::vector<std::shared_ptr<CallbackAbstract>> callbacks;
  
  // Check if it's a list
  if (PyList_Check(callbacks_list.ptr())) {
    bp::ssize_t n = PyList_Size(callbacks_list.ptr());
    for (bp::ssize_t i = 0; i < n; ++i) {
      PyObject* item = PyList_GetItem(callbacks_list.ptr(), i);
      bp::object item_obj(bp::handle<>(bp::borrowed(item)));
      bp::extract<std::shared_ptr<CallbackAbstract>> extractor(item_obj);
      if (extractor.check()) {
        callbacks.push_back(extractor());
      } else {
        PyErr_SetString(PyExc_TypeError, "Invalid callback object in list");
        bp::throw_error_already_set();
      }
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "setCallbacks expects a list");
    bp::throw_error_already_set();
  }
  self.setCallbacks(callbacks);
  return bp::object();  // Return None
}

void exposeSolverCSQP() {
  bp::register_ptr_to_python<std::shared_ptr<SolverCSQP> >();

  bp::class_<SolverCSQP, bp::bases<crocoddyl::SolverAbstract>, boost::noncopyable>(
      "SolverCSQP",
      "CSQP solver.\n\n"
      "The CSQP solver computes an optimal trajectory and control commands by "
      "iterates\n"
      "running SQP steps. Each iteration computes cost/constraint "
      "derivatives,\n"
      "solves the inner QP subproblem (via qp property), and performs line "
      "search.\n"
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
      .def("computeDirection", &SolverCSQP::computeDirection,
           bp::args("self", "recalcDiff"), "")
      .def("checkKKTConditions", &SolverCSQP::checkKKTConditions,
           bp::args("self"), "")

      // State/trial trajectories
      .def_readwrite("xs_try", &SolverCSQP::xs_try_, "xs try")
      .def_readwrite("us_try", &SolverCSQP::us_try_, "us try")
      .def_readwrite("cost_try", &SolverCSQP::cost_try_, "cost try")
      .def_readwrite("fs_try", &SolverCSQP::fs_try_, "fs_try")
      .def_readwrite("lag_mul", &SolverCSQP::lag_mul_, "lagrange multipliers")
      .def_readwrite("remove_reg", &SolverCSQP::remove_reg_,
                     "Removes Crocoddyl's regularization by setting "
                     "(preg,dreg)=0 when True (default: False)")

      // Inner QP solver access
      .add_property(
          "qp",
          bp::make_function(
              static_cast<SolverOSQP_QP& (SolverCSQP::*)()>(
                  &SolverCSQP::get_qp_solver),
              bp::return_internal_reference<>()),
          "Inner QP solver (SolverOSQP_QP). Use to configure ADMM parameters, "
          "e.g. solver.qp.sigma = 1e-5")

      // SQP-level parameters
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
      .add_property(
          "y",
          make_function(&SolverCSQP::get_y,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "ADMM dual variable")
      .add_property(
          "z",
          make_function(&SolverCSQP::get_z,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "ADMM z variable")
      .add_property(
          "rho_vec",
          make_function(&SolverCSQP::get_rho_vec,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Per-constraint rho values")

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

      .add_property("use_filter_line_search",
                    bp::make_function(&SolverCSQP::get_use_filter_line_search),
                    bp::make_function(&SolverCSQP::set_use_filter_line_search),
                    "Use the filter line search criteria (default: True)")
      .add_property(
          "termination_tolerance",
          bp::make_function(&SolverCSQP::get_termination_tolerance),
          bp::make_function(&SolverCSQP::set_termination_tolerance),
          "Termination criteria to exit the iteration (default: 1e-6)")
      .add_property("filter_size",
                    bp::make_function(&SolverCSQP::get_filter_size),
                    bp::make_function(&SolverCSQP::set_filter_size),
                    "filter size for the line-search (default: 1)")
      .add_property("max_solve_time",
                    bp::make_function(&SolverCSQP::get_max_solve_time),
                    bp::make_function(&SolverCSQP::set_max_solve_time),
                    "get and set max solve time in seconds")
      .add_property("max_solve_time_reached",
                    bp::make_function(&SolverCSQP::get_max_solve_time_reached),
                    "get if solver timed out")

      // Delegate to QP solver for backward compatibility
      .add_property("eps_abs", bp::make_function(&SolverCSQP::get_eps_abs),
                    bp::make_function(&SolverCSQP::set_eps_abs),
                    "Absolute termination criteria for QP solver (delegates to "
                    "qp.eps_abs)")
      .add_property("eps_rel", bp::make_function(&SolverCSQP::get_eps_rel),
                    bp::make_function(&SolverCSQP::set_eps_rel),
                    "Relative termination criteria for QP solver (delegates to "
                    "qp.eps_rel)")
      .add_property("rho_sparse",
                    bp::make_function(&SolverCSQP::get_rho_sparse),
                    bp::make_function(&SolverCSQP::set_rho_sparse),
                    "Rho value for QP solver (delegates to qp.rho_sparse)")
      .add_property(
          "equality_qp_initial_guess",
          bp::make_function(&SolverCSQP::get_equality_qp_initial_guess),
          bp::make_function(&SolverCSQP::set_equality_qp_initial_guess),
          "Initialize each QP with unconstrained solution (delegates to "
          "qp.equality_qp_initial_guess)")
      .add_property("sigma", bp::make_function(&SolverCSQP::get_sigma),
                    bp::make_function(&SolverCSQP::set_sigma),
                    "Proximal term (delegates to qp.sigma)")
      .add_property("alpha", bp::make_function(&SolverCSQP::get_alpha),
                    bp::make_function(&SolverCSQP::set_alpha),
                    "ADMM relaxation parameter (delegates to qp.alpha)")
      .add_property("max_qp_iters",
                    bp::make_function(&SolverCSQP::get_max_qp_iters),
                    bp::make_function(&SolverCSQP::set_max_qp_iters),
                    "Max QP iterations (delegates to qp.max_qp_iters)")
      .add_property("rho_update_interval",
                    bp::make_function(&SolverCSQP::get_rho_update_interval),
                    bp::make_function(&SolverCSQP::set_rho_update_interval),
                    "Rho update interval (delegates to qp.rho_update_interval)")
      .add_property("adaptive_rho_tolerance",
                    bp::make_function(&SolverCSQP::get_adaptive_rho_tolerance),
                    bp::make_function(&SolverCSQP::set_adaptive_rho_tolerance),
                    "Adaptive rho tolerance (delegates to "
                    "qp.adaptive_rho_tolerance)")
      .add_property("norm_primal",
                    bp::make_function(&SolverCSQP::get_norm_primal),
                    "Primal residual norm (from qp solver)")
      .add_property("norm_dual", bp::make_function(&SolverCSQP::get_norm_dual),
                    "Dual residual norm (from qp solver)")
      .add_property("norm_primal_rel",
                    bp::make_function(&SolverCSQP::get_norm_primal_rel),
                    "Relative primal residual norm (from qp solver)")
      .add_property("norm_dual_rel",
                    bp::make_function(&SolverCSQP::get_norm_dual_rel),
                    "Relative dual residual norm (from qp solver)")
      .add_property("norm_primal_tolerance",
                    bp::make_function(&SolverCSQP::get_norm_primal_tolerance),
                    "Primal tolerance (from qp solver)")
      .add_property("norm_dual_tolerance",
                    bp::make_function(&SolverCSQP::get_norm_dual_tolerance),
                    "Dual tolerance (from qp solver)")
      .add_property("reset_y", bp::make_function(&SolverCSQP::get_reset_y),
                    bp::make_function(&SolverCSQP::set_reset_y),
                    "Reset y between SQP iterations (delegates to qp.reset_y)")
      .add_property(
          "reset_rho", bp::make_function(&SolverCSQP::get_reset_rho),
          bp::make_function(&SolverCSQP::set_reset_rho),
          "Reset rho between SQP iterations (delegates to qp.reset_rho)")
      .add_property("with_qp_callbacks",
                    bp::make_function(&SolverCSQP::getQPCallbacks),
                    bp::make_function(&SolverCSQP::setQPCallbacks),
                    "Activates the QP callbacks when true (default: False)")

      // Override setCallbacks with a version that accepts mim_solvers callbacks
      .def("setCallbacks",
           bp::raw_function(&setCallbacks_wrapper),
           "Set callbacks for SQP solver iteration monitoring.\n\n"
           ":param callbacks: list of CallbackAbstract or derived objects "
           "(CallbackVerbose, CallbackLogger, etc.)")
      .def("getCallbacks",
           static_cast<const std::vector<std::shared_ptr<CallbackAbstract>>&
               (SolverCSQP::*)() const>(&SolverCSQP::getCallbacks),
           bp::return_value_policy<bp::copy_const_reference>(),
           "Get the callbacks");
}

}  // namespace mim_solvers
