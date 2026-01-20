///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_CSQP_HPP_
#define MIM_SOLVERS_CSQP_HPP_

#include <Eigen/Cholesky>
#include <boost/circular_buffer.hpp>
#include <limits>
#include <memory>
#include <vector>

#include "mim_solvers/ddp.hpp"
#include "mim_solvers/osqp_qp.hpp"
#include "mim_solvers/utils/callbacks.hpp"

namespace mim_solvers {

/**
 * @brief Constrained Sequential Quadratic Programming (CSQP) solver
 *
 * The CSQP solver computes an optimal trajectory and control commands by
 * iterating SQP steps. Each iteration:
 * 1. Compute cost/constraints residuals and derivatives (calc)
 * 2. Solve the inner QP subproblem (computeDirection via SolverOSQP_QP)
 * 3. Perform merit/filter line search (tryStep)
 * 4. Check KKT conditions
 *
 * The QP solver is encapsulated in the SolverOSQP_QP class, which uses
 * an ADMM algorithm exploiting the temporal sparsity of optimal control.
 *
 * SolverCSQP is now independent of SolverDDP, inheriting directly from
 * crocoddyl::SolverAbstract. All DDP-related data (Vxx, Qxx, etc.) is now
 * managed by the internal SolverOSQP_QP solver.
 *
 * \sa `SolverOSQP_QP()`
 */
class SolverCSQP : public crocoddyl::SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the CSQP solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverCSQP(std::shared_ptr<crocoddyl::ShootingProblem> problem);
  virtual ~SolverCSQP();

  virtual bool solve(
      const std::vector<Eigen::VectorXd>& init_xs = crocoddyl::DEFAULT_VECTOR,
      const std::vector<Eigen::VectorXd>& init_us = crocoddyl::DEFAULT_VECTOR,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const double regInit = NAN);

  /**
   * @brief Computes the merit function, gaps at the given xs, us along with
   * delta x and delta u
   */
  virtual void computeDirection(const bool recalcDiff);

  virtual double tryStep(const double stepLength);

  virtual void calc(const bool recalc = true);

  /**
   * @brief Compute the stopping criteria
   */
  virtual double stoppingCriteria();

  /**
   * @brief Get expected improvement
   */
  virtual const Eigen::Vector2d& expectedImprovement();

  /**
   * @brief Compute the KKT conditions residual
   */
  virtual void checkKKTConditions();

  // ========================
  // Getters for state/outputs
  // ========================
  const std::vector<Eigen::VectorXd>& get_xs_try() const { return xs_try_; }
  const std::vector<Eigen::VectorXd>& get_us_try() const { return us_try_; }

  const std::vector<Eigen::VectorXd>& get_xs() const { return xs_; }
  const std::vector<Eigen::VectorXd>& get_us() const { return us_; }
  const std::vector<Eigen::VectorXd>& get_fs() const { return qp_solver_->get_fs(); }

  // Delegate to QP solver
  const std::vector<Eigen::VectorXd>& get_dx_tilde() const {
    return qp_solver_->get_dx_tilde();
  }
  const std::vector<Eigen::VectorXd>& get_du_tilde() const {
    return qp_solver_->get_du_tilde();
  }
  const std::vector<Eigen::VectorXd>& get_dx() const {
    return qp_solver_->get_dx();
  }
  const std::vector<Eigen::VectorXd>& get_du() const {
    return qp_solver_->get_du();
  }
  const std::vector<Eigen::VectorXd>& get_y() const {
    return qp_solver_->get_y();
  }
  const std::vector<Eigen::VectorXd>& get_z() const {
    return qp_solver_->get_z();
  }
  const std::vector<Eigen::VectorXd>& get_rho_vec() const {
    return qp_solver_->get_rho_vec();
  }

  double get_KKT() const { return KKT_; }
  double get_gap_norm() const { return gap_norm_; }
  double get_constraint_norm() const { return constraint_norm_; }
  double get_qp_iters() const { return qp_solver_->get_qp_iters(); }
  double get_xgrad_norm() const { return x_grad_norm_; }
  double get_ugrad_norm() const { return u_grad_norm_; }
  double get_merit() const { return merit_; }

  bool get_extra_iteration_for_last_kkt() const {
    return extra_iteration_for_last_kkt_;
  }
  bool get_use_filter_line_search() const { return use_filter_line_search_; }
  double get_lag_mul_inf_norm_coef() const { return lag_mul_inf_norm_coef_; }
  double get_mu_dynamic() const { return mu_dynamic_; }
  double get_mu_constraint() const { return mu_constraint_; }
  double get_termination_tolerance() const { return termination_tol_; }
  std::size_t get_filter_size() const { return filter_size_; }

  // Delegate QP parameters to qp_solver_
  std::size_t get_max_qp_iters() const { return qp_solver_->get_max_qp_iters(); }
  bool get_equality_qp_initial_guess() const {
    return qp_solver_->get_equality_qp_initial_guess();
  }
  std::size_t get_rho_update_interval() const {
    return qp_solver_->get_rho_update_interval();
  }
  double get_adaptive_rho_tolerance() const {
    return qp_solver_->get_adaptive_rho_tolerance();
  }
  double get_alpha() const { return qp_solver_->get_alpha(); }
  double get_sigma() const { return qp_solver_->get_sigma(); }
  double get_rho_sparse() const { return qp_solver_->get_rho_sparse(); }
  double get_eps_abs() const { return qp_solver_->get_eps_abs(); }
  double get_eps_rel() const { return qp_solver_->get_eps_rel(); }
  double get_norm_primal() const { return qp_solver_->get_norm_primal(); }
  double get_norm_primal_tolerance() const {
    return qp_solver_->get_norm_primal_tolerance();
  }
  double get_norm_dual() const { return qp_solver_->get_norm_dual(); }
  double get_norm_primal_rel() const {
    return qp_solver_->get_norm_primal_rel();
  }
  double get_norm_dual_rel() const { return qp_solver_->get_norm_dual_rel(); }
  double get_norm_dual_tolerance() const {
    return qp_solver_->get_norm_dual_tolerance();
  }
  bool get_reset_y() const { return qp_solver_->get_reset_y(); }
  bool get_reset_rho() const { return qp_solver_->get_reset_rho(); }
  double get_rho_min() const { return qp_solver_->get_rho_min(); }
  double get_rho_max() const { return qp_solver_->get_rho_max(); }
  double get_max_solve_time() const { return max_solve_time_; }
  bool get_max_solve_time_reached() const { return max_solve_time_reached_; }

  bool getQPCallbacks() const { return qp_solver_->getQPCallbacks(); }

  // ========================
  // Setters
  // ========================
  void setQPCallbacks(const bool inCallbacks) {
    qp_solver_->setQPCallbacks(inCallbacks);
  }

  void set_rho_update_interval(const std::size_t interval) {
    qp_solver_->set_rho_update_interval(interval);
  }
  void set_adaptive_rho_tolerance(const double tolerance) {
    qp_solver_->set_adaptive_rho_tolerance(tolerance);
  }

  void set_lag_mul_inf_norm_coef(const double lag_mul_inf_norm_coef) {
    lag_mul_inf_norm_coef_ = lag_mul_inf_norm_coef;
  }
  void set_mu_dynamic(const double mu_dynamic) { mu_dynamic_ = mu_dynamic; }
  void set_mu_constraint(const double mu_constraint) {
    mu_constraint_ = mu_constraint;
  }
  void set_alpha(const double alpha) { qp_solver_->set_alpha(alpha); }
  void set_sigma(const double sigma) { qp_solver_->set_sigma(sigma); }

  void set_equality_qp_initial_guess(const bool equality_qp_initial_guess) {
    qp_solver_->set_equality_qp_initial_guess(equality_qp_initial_guess);
  }

  void set_termination_tolerance(const double tol) { termination_tol_ = tol; }
  void set_extra_iteration_for_last_kkt(const bool inBool) {
    extra_iteration_for_last_kkt_ = inBool;
  }
  void set_use_filter_line_search(const bool inBool) {
    use_filter_line_search_ = inBool;
  }
  void set_filter_size(const std::size_t inFilterSize) {
    filter_size_ = inFilterSize;
    gap_list_.resize(filter_size_);
    constraint_list_.resize(filter_size_);
    cost_list_.resize(filter_size_);
  }

  void set_rho_sparse(const double rho_sparse) {
    qp_solver_->set_rho_sparse(rho_sparse);
  }

  void set_max_qp_iters(const std::size_t iters) {
    qp_solver_->set_max_qp_iters(iters);
  }
  void set_eps_abs(const double eps_abs) { qp_solver_->set_eps_abs(eps_abs); }
  void set_eps_rel(const double eps_rel) { qp_solver_->set_eps_rel(eps_rel); }
  void set_max_solve_time(const double max_solve_time) {
    max_solve_time_ = max_solve_time;
  }

  void set_reset_y(const bool val) { qp_solver_->set_reset_y(val); }
  void set_reset_rho(const bool val) { qp_solver_->set_reset_rho(val); }

  /**
   * @brief Set callbacks for solver iteration monitoring (mim_solvers callbacks)
   *
   * @param[in] callbacks vector of mim_solvers callback objects
   */
  void setCallbacks(
      const std::vector<std::shared_ptr<CallbackAbstract>>& callbacks) {
    callbacks_ = callbacks;
  }

  /**
   * @brief Get the callbacks
   */
  const std::vector<std::shared_ptr<CallbackAbstract>>& getCallbacks() const {
    return callbacks_;
  }

  /**
   * @brief Get access to the inner QP solver for advanced configuration
   */
  SolverOSQP_QP& get_qp_solver() { return *qp_solver_; }
  const SolverOSQP_QP& get_qp_solver() const { return *qp_solver_; }

 public:
  // Filter line search buffers
  boost::circular_buffer<double> constraint_list_;
  boost::circular_buffer<double> gap_list_;
  boost::circular_buffer<double> cost_list_;

  std::vector<Eigen::VectorXd> fs_try_;  //!< Gaps/defects for trial trajectory
  std::vector<Eigen::VectorXd>
      lag_mul_;  //!< Lagrange multipliers for dynamics constraint
  std::vector<Eigen::VectorXd> xs_try_;  //!< Trial state trajectory
  std::vector<Eigen::VectorXd> us_try_;  //!< Trial control trajectory

  double lag_mul_inf_norm_;  //!< Infinity norm of Lagrange multipliers
  double lag_mul_inf_norm_coef_ = 10.;  //!< Merit function coefficient
  Eigen::VectorXd fs_flat_;  //!< Gaps/defects as 1D array

  bool use_filter_line_search_ = true;  //!< Use filter line search

  // Keep public access to QP solver ADMM variables for compatibility
  bool remove_reg_ = false;  //!< Remove Crocoddyl's regularization

  // Cost and merit function values (for Python bindings)
  double cost_try_ = 0;         //!< Cost function for trial step

 protected:
  // SQP-specific state
  // ========================
  std::unique_ptr<SolverOSQP_QP> qp_solver_;  //!< Inner QP solver

  double merit_ = 0;            //!< Merit function at nominal trajectory
  double merit_try_ = 0;        //!< Merit function for trial step
  double x_grad_norm_ = 0;      //!< 1-norm of delta x
  double u_grad_norm_ = 0;      //!< 1-norm of delta u
  double gap_norm_ = 0;         //!< 1-norm of gaps
  double constraint_norm_ = 0;  //!< 1-norm of constraint violation
  double constraint_norm_try_ = 0;  //!< Constraint violation for trial
  double gap_norm_try_ = 0;     //!< Gap norm for trial
  double mu_dynamic_ = 1e1;     //!< Merit penalty for dynamics violation
  double mu_constraint_ = 1e1;  //!< Merit penalty for constraint violation
  double termination_tol_ = 1e-6;  //!< Termination tolerance for KKT

  bool extra_iteration_for_last_kkt_ = false;
  std::size_t filter_size_ = 1;  //!< Filter size for line search
  double KKT_ = std::numeric_limits<double>::infinity();
  Eigen::Vector2d expected_improvement_ = Eigen::Vector2d::Zero();  //!< Expected improvement

  std::vector<std::shared_ptr<CallbackAbstract>>
      callbacks_;  //!< Callbacks for iteration monitoring

protected:
  /**
   * @brief Increase state and control regularization values
   */
  void increaseRegularization();

  /**
   * @brief Decrease state and control regularization values
   */
  void decreaseRegularization();

private:
  double th_acceptnegstep_;
  double th_stepdec_ = 0.1;     //!< Step length threshold for decreasing filter
  bool is_worse_than_memory_ = false;

  Eigen::VectorXd tmp_vec_x_;
  std::vector<Eigen::VectorXd> tmp_vec_u_;

  double start_time_ = 0.0;
  bool max_solve_time_reached_ = false;
  double max_solve_time_ = std::numeric_limits<double>::infinity();
  std::vector<double> alphas_;      //!< Step lengths for line search
};  // class SolverCSQP

}  // namespace mim_solvers

#endif  // MIM_SOLVERS_CSQP_HPP_
