///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_OSQP_QP_HPP_
#define MIM_SOLVERS_OSQP_QP_HPP_

#include <Eigen/Cholesky>
#include <limits>
#include <vector>

#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/mathbase.hpp>

namespace mim_solvers {

// Type alias for row-major matrices (matching SolverDDP)
typedef typename crocoddyl::MathBaseTpl<double>::MatrixXsRowMajor
    MatrixXdRowMajor;

/**
 * @brief OSQP-style QP solver for optimal control problems
 *
 * This class implements an ADMM-based QP solver that exploits the temporal
 * sparsity of optimal control problems. It solves the constrained QP subproblem
 * arising in SQP methods by performing LQR backward/forward passes with
 * projections to enforce constraints.
 *
 * The solver takes as input the problem's data (partial derivatives of cost
 * and constraints) and returns the primal (dx, du) and dual (y) solutions.
 *
 * This class is designed to be used as an inner solver within SolverCSQP,
 * but can also be used standalone for custom SQP implementations.
 */
class SolverOSQP_QP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the OSQP_QP solver
   *
   * @param[in] problem  Shooting problem (for dimensions and constraint bounds)
   */
  explicit SolverOSQP_QP(std::shared_ptr<crocoddyl::ShootingProblem> problem);
  virtual ~SolverOSQP_QP();

  /**
   * @brief Solve the QP to compute search direction (dx, du)
   *
   * Runs the ADMM loop: backward pass, forward pass, dual update, rho update.
   * Terminates when primal/dual residuals are below tolerance or max iterations.
   */
  virtual void computeDirection();

  /**
   * @brief Reset ADMM parameters between SQP iterations
   *
   * Resets z, z_prev, z_relaxed. Optionally resets y and rho based on flags.
   */
  virtual void reset_params();

  /**
   * @brief Reset rho vector to base value
   */
  virtual void reset_rho_vec();

  // Backward pass variants
  virtual void backwardPass();
  virtual void backwardPass_without_rho_update();
  virtual void backwardPass_without_constraints();

#ifdef CROCODDYL_WITH_MULTITHREADING
  virtual void backwardPass_mt();
  virtual void backwardPass_without_rho_update_mt();
#endif

  // Forward pass variants
  virtual void forwardPass();
  virtual void forwardPass_without_constraints();

  /**
   * @brief Update ADMM dual variables (z projection and y update)
   */
  virtual void update_lagrangian_parameters(const int iter);

  /**
   * @brief Update rho vector adaptively based on residuals
   */
  virtual void update_rho_vec(const int iter);

  /**
   * @brief Apply rho update to all constraints
   */
  virtual void apply_rho_update(const double rho_sparse);

  /**
   * @brief Compute the feedforward and feedback terms using Cholesky decomposition
   *
   * Computes k and K from Quu, Qu, Qxu using LLT factorization.
   */
  virtual void computeGains(const std::size_t t);

  /**
   * @brief Increase state and control regularization values
   */
  void increaseRegularization();

  /**
   * @brief Allocate internal data structures for the QP solver
   */
  void allocateData();

  /**
   * @brief Print QP iteration information
   */
  void printQPCallbacks(const int iter);

  /**
   * @brief Enable/disable QP iteration logging
   */
  void setQPCallbacks(const bool inCallbacks);

  // ========================
  // Getters for outputs
  // ========================
  const std::vector<Eigen::VectorXd>& get_dx() const { return dx_; }
  const std::vector<Eigen::VectorXd>& get_du() const { return du_; }
  const std::vector<Eigen::VectorXd>& get_dx_tilde() const { return dxtilde_; }
  const std::vector<Eigen::VectorXd>& get_du_tilde() const { return dutilde_; }
  const std::vector<Eigen::VectorXd>& get_y() const { return y_; }
  const std::vector<Eigen::VectorXd>& get_z() const { return z_; }
  const std::vector<Eigen::VectorXd>& get_rho_vec() const { return rho_vec_; }

  std::size_t get_qp_iters() const { return qp_iters_; }
  double get_norm_primal() const { return norm_primal_; }
  double get_norm_dual() const { return norm_dual_; }
  double get_norm_primal_rel() const { return norm_primal_rel_; }
  double get_norm_dual_rel() const { return norm_dual_rel_; }
  double get_norm_primal_tolerance() const { return norm_primal_tolerance_; }
  double get_norm_dual_tolerance() const { return norm_dual_tolerance_; }
  bool getQPCallbacks() const { return with_qp_callbacks_; }

  // ========================
  // Getters/Setters for ADMM parameters
  // ========================
  double get_sigma() const { return sigma_; }
  void set_sigma(const double sigma) { sigma_ = sigma; }

  double get_alpha() const { return alpha_; }
  void set_alpha(const double alpha) { alpha_ = alpha; }

  void set_rho_sparse(const double rho_sparse) { rho_sparse_ = rho_sparse; }
  double get_rho_sparse() const { return rho_sparse_; }

  double get_rho_min() const { return rho_min_; }
  double get_rho_max() const { return rho_max_; }

  std::size_t get_rho_update_interval() const { return rho_update_interval_; }
  void set_rho_update_interval(const std::size_t interval) {
    rho_update_interval_ = interval;
  }

  double get_adaptive_rho_tolerance() const { return adaptive_rho_tolerance_; }
  void set_adaptive_rho_tolerance(const double tolerance) {
    adaptive_rho_tolerance_ = tolerance;
  }

  double get_eps_abs() const { return eps_abs_; }
  void set_eps_abs(const double eps_abs) { eps_abs_ = eps_abs; }

  double get_eps_rel() const { return eps_rel_; }
  void set_eps_rel(const double eps_rel) { eps_rel_ = eps_rel; }

  std::size_t get_max_qp_iters() const { return max_qp_iters_; }
  void set_max_qp_iters(const std::size_t iters) { max_qp_iters_ = iters; }

  bool get_equality_qp_initial_guess() const {
    return equality_qp_initial_guess_;
  }
  void set_equality_qp_initial_guess(const bool val) {
    equality_qp_initial_guess_ = val;
  }

  bool get_reset_y() const { return reset_y_; }
  void set_reset_y(const bool val) { reset_y_ = val; }

  bool get_reset_rho() const { return reset_rho_; }
  void set_reset_rho(const bool val) { reset_rho_ = val; }

  bool get_update_rho_with_heuristic() const {
    return update_rho_with_heuristic_;
  }
  void set_update_rho_with_heuristic(const bool val) {
    update_rho_with_heuristic_ = val;
  }

  // For timeout coordination with parent solver
  void set_start_time(double t) { start_time_ = t; }
  void set_max_solve_time(double t) { max_solve_time_ = t; }
  bool get_max_solve_time_reached() const { return max_solve_time_reached_; }

  // Regularization getters/setters
  double get_preg() const { return preg_; }
  void set_preg(const double preg) { preg_ = preg; }
  
  double get_dreg() const { return dreg_; }
  void set_dreg(const double dreg) { dreg_ = dreg; }
  
  double get_reg_min() const { return reg_min_; }
  double get_reg_max() const { return reg_max_; }
  double get_reg_incfactor() const { return reg_incfactor_; }
  double get_reg_decfactor() const { return reg_decfactor_; }

  // DDP data getters (for advanced users)
  const std::vector<Eigen::MatrixXd>& get_Vxx() const { return Vxx_; }
  const std::vector<Eigen::VectorXd>& get_Vx() const { return Vx_; }
  const std::vector<Eigen::MatrixXd>& get_Qxx() const { return Qxx_; }
  const std::vector<Eigen::MatrixXd>& get_Qxu() const { return Qxu_; }
  const std::vector<Eigen::MatrixXd>& get_Quu() const { return Quu_; }
  const std::vector<Eigen::VectorXd>& get_Qx() const { return Qx_; }
  const std::vector<Eigen::VectorXd>& get_Qu() const { return Qu_; }
  const std::vector<MatrixXdRowMajor>& get_K() const { return K_; }
  const std::vector<Eigen::VectorXd>& get_k() const { return k_; }

 public:
  // ========================
  // ADMM Variables (public for direct access from bindings)
  // ========================
  std::vector<Eigen::VectorXd> dx_;       //!< Descent direction for x (output)
  std::vector<Eigen::VectorXd> du_;       //!< Descent direction for u (output)
  std::vector<Eigen::VectorXd> dxtilde_;  //!< ADMM x-tilde variable
  std::vector<Eigen::VectorXd> dutilde_;  //!< ADMM u-tilde variable
  std::vector<Eigen::VectorXd> fs_;       //!< Dynamics gaps

  std::vector<Eigen::VectorXd> y_;            //!< ADMM dual variable
  std::vector<Eigen::VectorXd> z_;            //!< ADMM z variable
  std::vector<Eigen::VectorXd> z_prev_;       //!< Previous z (for dual residual)
  std::vector<Eigen::VectorXd> z_relaxed_;    //!< Over-relaxed z step
  std::vector<Eigen::VectorXd> rho_vec_;      //!< Per-constraint rho values
  std::vector<Eigen::VectorXd> inv_rho_vec_;  //!< Inverse rho for efficiency

  double norm_primal_ = 0.0;      //!< Primal residual norm
  double norm_dual_ = 0.0;        //!< Dual residual norm
  double norm_primal_rel_ = 0.0;  //!< Relative primal residual
  double norm_dual_rel_ = 0.0;    //!< Relative dual residual
  double norm_primal_tolerance_ = 0.0;  //!< Primal convergence tolerance
  double norm_dual_tolerance_ = 0.0;    //!< Dual convergence tolerance

  bool reset_y_ = false;                  //!< Reset y between SQP iterations
  bool reset_rho_ = false;                //!< Reset rho between SQP iterations
  bool update_rho_with_heuristic_ = false;  //!< Use heuristic for rho update

 protected:
  std::shared_ptr<crocoddyl::ShootingProblem> problem_;  //!< Shooting problem

  // ========================
  // DDP Data (Value Function, Hamiltonian, and Gains)
  // ========================
  std::vector<Eigen::MatrixXd>
      Vxx_;  //!< Hessian of the Value function \f$\mathbf{V_{xx}}\f$
  Eigen::MatrixXd
      Vxx_tmp_;  //!< Temporary variable for ensuring symmetry of Vxx
  std::vector<Eigen::VectorXd>
      Vx_;  //!< Gradient of the Value function \f$\mathbf{V_x}\f$
  std::vector<Eigen::MatrixXd>
      Qxx_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{xx}}\f$
  std::vector<Eigen::MatrixXd>
      Qxu_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{xu}}\f$
  std::vector<Eigen::MatrixXd>
      Quu_;  //!< Hessian of the Hamiltonian \f$\mathbf{Q_{uu}}\f$
  std::vector<Eigen::VectorXd>
      Qx_;  //!< Gradient of the Hamiltonian \f$\mathbf{Q_x}\f$
  std::vector<Eigen::VectorXd>
      Qu_;  //!< Gradient of the Hamiltonian \f$\mathbf{Q_u}\f$
  std::vector<MatrixXdRowMajor> K_;  //!< Feedback gains \f$\mathbf{K}\f$
  std::vector<Eigen::VectorXd> k_;   //!< Feed-forward terms \f$\mathbf{k}\f$

  // Temporary/working data for backward pass
  Eigen::VectorXd xnext_;      //!< Next state \f$\mathbf{x}^{'}\f$
  MatrixXdRowMajor FxTVxx_p_;  //!< Store the value of
                               //!< \f$\mathbf{f_x}^T\mathbf{V_{xx}}^{'}\f$
  std::vector<MatrixXdRowMajor>
      FuTVxx_p_;             //!< Store the values of
                             //!< \f$\mathbf{f_u}^T\mathbf{V_{xx}}^{'}\f$
                             //!< per each running node
  Eigen::VectorXd fTVxx_p_;  //!< Store the value of
                             //!< \f$\mathbf{\bar{f}}^T\mathbf{V_{xx}}^{'}\f$
  std::vector<Eigen::LLT<Eigen::MatrixXd> > Quu_llt_;  //!< Cholesky LLT solver
  std::vector<Eigen::VectorXd>
      Quuk_;  //!< Store the values of \f$\mathbf{Q_{uu}\mathbf{k}}\f$

  // ========================
  // Regularization Parameters (copied from SolverDDP)
  // ========================
  double preg_ = 0.0;           //!< State regularization
  double dreg_ = 0.0;           //!< Control regularization
  double reg_min_ = 1e-9;       //!< Minimum regularization value
  double reg_max_ = 1e9;        //!< Maximum regularization value
  double reg_incfactor_ = 10.0; //!< Factor to increase regularization
  double reg_decfactor_ = 10.0; //!< Factor to decrease regularization

  // ========================
  // ADMM Parameters
  // ========================
  double sigma_ = 1e-6;   //!< Proximal term
  double alpha_ = 1.6;    //!< ADMM relaxation parameter (over-relaxation)

  double rho_sparse_;             //!< Current rho value
  double rho_sparse_base_ = 1e-1; //!< Base rho value
  double rho_min_ = 1e-6;         //!< Minimum rho
  double rho_max_ = 1e3;          //!< Maximum rho
  double rho_estimate_sparse_ = 0.0;  //!< Estimated optimal rho

  std::size_t rho_update_interval_ = 25;  //!< Frequency of rho update
  double adaptive_rho_tolerance_ = 5;     //!< Threshold for rho update

  double eps_abs_ = 1e-4;  //!< Absolute convergence tolerance
  double eps_rel_ = 1e-4;  //!< Relative convergence tolerance

  std::size_t max_qp_iters_ = 1000;  //!< Maximum QP iterations
  std::size_t qp_iters_ = 0;         //!< Actual QP iterations used

  bool equality_qp_initial_guess_ = true;  //!< Warm-start with unconstrained solution
  bool with_qp_callbacks_ = false;         //!< Enable QP iteration logging

  // Timeout coordination
  double start_time_ = 0.0;
  double max_solve_time_ = std::numeric_limits<double>::infinity();
  bool max_solve_time_reached_ = false;

 private:
  // Temporary variables for computation
  Eigen::VectorXd tmp_vec_x_;
  std::vector<Eigen::VectorXd> tmp_vec_u_;
  std::vector<Eigen::VectorXd> tmp_dual_cwise_;
  Eigen::VectorXd tmp_Vx_;
  std::vector<Eigen::VectorXd> tmp_Cdx_Cdu_;
  std::vector<Eigen::MatrixXd> tmp_rhoGx_mat_;
  std::vector<Eigen::MatrixXd> tmp_rhoGu_mat_;
  std::vector<Eigen::VectorXd> Vxx_fs_;
};

}  // namespace mim_solvers

#endif  // MIM_SOLVERS_OSQP_QP_HPP_
