///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mim_solvers/osqp_qp.hpp"

#include "mim_solvers/python.hpp"

namespace mim_solvers {

namespace bp = boost::python;

void exposeSolverOSQP_QP() {
  bp::register_ptr_to_python<std::shared_ptr<SolverOSQP_QP>>();

  bp::class_<SolverOSQP_QP, boost::noncopyable>(
      "SolverOSQP_QP",
      "OSQP-style QP solver for optimal control problems.\n\n"
      "This class implements an ADMM-based QP solver that exploits the temporal\n"
      "sparsity of optimal control problems. It is typically used as the inner\n"
      "solver within SolverCSQP, accessible via the 'qp' property.",
      bp::no_init)  // No direct construction from Python for now
      
      // ADMM parameters
      .add_property("sigma", bp::make_function(&SolverOSQP_QP::get_sigma),
                    bp::make_function(&SolverOSQP_QP::set_sigma),
                    "Proximal term (default: 1e-6)")
      .add_property("alpha", bp::make_function(&SolverOSQP_QP::get_alpha),
                    bp::make_function(&SolverOSQP_QP::set_alpha),
                    "ADMM relaxation parameter (default: 1.6)")
      .add_property("rho_sparse",
                    bp::make_function(&SolverOSQP_QP::get_rho_sparse),
                    bp::make_function(&SolverOSQP_QP::set_rho_sparse),
                    "Current rho value (default: 0.1)")
      .add_property("rho_min", bp::make_function(&SolverOSQP_QP::get_rho_min),
                    "Minimum rho value")
      .add_property("rho_max", bp::make_function(&SolverOSQP_QP::get_rho_max),
                    "Maximum rho value")
      .add_property("rho_update_interval",
                    bp::make_function(&SolverOSQP_QP::get_rho_update_interval),
                    bp::make_function(&SolverOSQP_QP::set_rho_update_interval),
                    "Frequency of adaptive rho update (default: 25)")
      .add_property("adaptive_rho_tolerance",
                    bp::make_function(&SolverOSQP_QP::get_adaptive_rho_tolerance),
                    bp::make_function(&SolverOSQP_QP::set_adaptive_rho_tolerance),
                    "Threshold for rho update (default: 5)")
      
      // Convergence parameters
      .add_property("eps_abs", bp::make_function(&SolverOSQP_QP::get_eps_abs),
                    bp::make_function(&SolverOSQP_QP::set_eps_abs),
                    "Absolute convergence tolerance (default: 1e-4)")
      .add_property("eps_rel", bp::make_function(&SolverOSQP_QP::get_eps_rel),
                    bp::make_function(&SolverOSQP_QP::set_eps_rel),
                    "Relative convergence tolerance (default: 1e-4)")
      .add_property("max_qp_iters",
                    bp::make_function(&SolverOSQP_QP::get_max_qp_iters),
                    bp::make_function(&SolverOSQP_QP::set_max_qp_iters),
                    "Maximum QP iterations (default: 1000)")
      .add_property("equality_qp_initial_guess",
                    bp::make_function(&SolverOSQP_QP::get_equality_qp_initial_guess),
                    bp::make_function(&SolverOSQP_QP::set_equality_qp_initial_guess),
                    "Warm-start with unconstrained solution (default: True)")
      
      // Reset flags
      .add_property("reset_y",
                    bp::make_function(&SolverOSQP_QP::get_reset_y),
                    bp::make_function(&SolverOSQP_QP::set_reset_y),
                    "Reset ADMM dual variable y between SQP iterations (default: False)")
      .add_property("reset_rho",
                    bp::make_function(&SolverOSQP_QP::get_reset_rho),
                    bp::make_function(&SolverOSQP_QP::set_reset_rho),
                    "Reset rho between SQP iterations (default: False)")
      .add_property("update_rho_with_heuristic",
                    bp::make_function(&SolverOSQP_QP::get_update_rho_with_heuristic),
                    bp::make_function(&SolverOSQP_QP::set_update_rho_with_heuristic),
                    "Use heuristic for rho update (default: False)")
      
      // Callback
      .add_property("with_qp_callbacks",
                    bp::make_function(&SolverOSQP_QP::getQPCallbacks),
                    bp::make_function(&SolverOSQP_QP::setQPCallbacks),
                    "Enable QP iteration logging (default: False)")
      
      // Read-only outputs
      .add_property("qp_iters", bp::make_function(&SolverOSQP_QP::get_qp_iters),
                    "Number of QP iterations used in last solve")
      .add_property("norm_primal", bp::make_function(&SolverOSQP_QP::get_norm_primal),
                    "Primal residual norm")
      .add_property("norm_dual", bp::make_function(&SolverOSQP_QP::get_norm_dual),
                    "Dual residual norm")
      .add_property("norm_primal_rel",
                    bp::make_function(&SolverOSQP_QP::get_norm_primal_rel),
                    "Relative primal residual norm")
      .add_property("norm_dual_rel",
                    bp::make_function(&SolverOSQP_QP::get_norm_dual_rel),
                    "Relative dual residual norm")
      .add_property("norm_primal_tolerance",
                    bp::make_function(&SolverOSQP_QP::get_norm_primal_tolerance),
                    "Primal convergence tolerance")
      .add_property("norm_dual_tolerance",
                    bp::make_function(&SolverOSQP_QP::get_norm_dual_tolerance),
                    "Dual convergence tolerance")
      
      // Solution vectors (read-only)
      .add_property(
          "dx",
          make_function(&SolverOSQP_QP::get_dx,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Descent direction for states")
      .add_property(
          "du",
          make_function(&SolverOSQP_QP::get_du,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Descent direction for controls")
      .add_property(
          "dx_tilde",
          make_function(&SolverOSQP_QP::get_dx_tilde,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "ADMM x-tilde variable")
      .add_property(
          "du_tilde",
          make_function(&SolverOSQP_QP::get_du_tilde,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "ADMM u-tilde variable")
      .add_property(
          "y",
          make_function(&SolverOSQP_QP::get_y,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "ADMM dual variable")
      .add_property(
          "z",
          make_function(&SolverOSQP_QP::get_z,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "ADMM z variable")
      .add_property(
          "rho_vec",
          make_function(&SolverOSQP_QP::get_rho_vec,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Per-constraint rho values")
      
      // Regularization parameters
      .add_property("preg", bp::make_function(&SolverOSQP_QP::get_preg),
                    "State regularization value")
      .add_property("dreg", bp::make_function(&SolverOSQP_QP::get_dreg),
                    "Control regularization value")
      .add_property("reg_min", bp::make_function(&SolverOSQP_QP::get_reg_min),
                    "Minimum regularization value (default: 1e-9)")
      .add_property("reg_max", bp::make_function(&SolverOSQP_QP::get_reg_max),
                    "Maximum regularization value (default: 1e9)")
      .add_property("reg_incfactor", bp::make_function(&SolverOSQP_QP::get_reg_incfactor),
                    "Factor to increase regularization (default: 10.0)")
      .add_property("reg_decfactor", bp::make_function(&SolverOSQP_QP::get_reg_decfactor),
                    "Factor to decrease regularization (default: 10.0)")
      
      // DDP data - Value function
      .add_property(
          "Vxx",
          make_function(&SolverOSQP_QP::get_Vxx,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Hessian of the value function (read-only)")
      .add_property(
          "Vx",
          make_function(&SolverOSQP_QP::get_Vx,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Gradient of the value function (read-only)")
      
      // DDP data - Hamiltonian/Q-function
      .add_property(
          "Qxx",
          make_function(&SolverOSQP_QP::get_Qxx,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Hessian of the Hamiltonian w.r.t. state (read-only)")
      .add_property(
          "Qxu",
          make_function(&SolverOSQP_QP::get_Qxu,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Hessian of the Hamiltonian w.r.t. state and control (read-only)")
      .add_property(
          "Quu",
          make_function(&SolverOSQP_QP::get_Quu,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Hessian of the Hamiltonian w.r.t. control (read-only)")
      .add_property(
          "Qx",
          make_function(&SolverOSQP_QP::get_Qx,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Gradient of the Hamiltonian w.r.t. state (read-only)")
      .add_property(
          "Qu",
          make_function(&SolverOSQP_QP::get_Qu,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Gradient of the Hamiltonian w.r.t. control (read-only)")
      
      // DDP data - Gains
      .add_property(
          "K",
          make_function(&SolverOSQP_QP::get_K,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Feedback gains (read-only)")
      .add_property(
          "k",
          make_function(&SolverOSQP_QP::get_k,
                        bp::return_value_policy<bp::copy_const_reference>()),
          "Feedforward terms (read-only)");
}

}  // namespace mim_solvers
