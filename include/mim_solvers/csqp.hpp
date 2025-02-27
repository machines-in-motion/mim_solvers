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
#include <vector>

#include "mim_solvers/ddp.hpp"
#include "mim_solvers/utils/callbacks.hpp"

namespace mim_solvers {

/**
 * @brief Constrained Sequential Quadratic Programming (CSQP) solver
 *
 * The CSQP solver computes an optimal trajectory and control commands by
 * iterates running `backwardPass()` and `forwardPass()`. The backward pass
 * accepts infeasible guess as described in the `SolverDDP::backwardPass()`.
 * Additionally, the forward pass handles infeasibility simulations that
 * resembles the numerical behaviour of a multiple-shooting formulation, i.e.:
 * \f{eqnarray}
 *   \mathbf{\hat{x}}_0 &=& \mathbf{\tilde{x}}_0 - (1 -
 * \alpha)\mathbf{\bar{f}}_0,\\
 *   \mathbf{\hat{u}}_k &=& \mathbf{u}_k + \alpha\mathbf{k}_k +
 * \mathbf{K}_k(\mathbf{\hat{x}}_k-\mathbf{x}_k),\\
 *   \mathbf{\hat{x}}_{k+1} &=&
 * \mathbf{f}_k(\mathbf{\hat{x}}_k,\mathbf{\hat{u}}_k) - (1 -
 * \alpha)\mathbf{\bar{f}}_{k+1}.
 * \f}
 * Note that the forward pass keeps the gaps \f$\mathbf{\bar{f}}_s\f$ open
 * according to the step length \f$\alpha\f$ that has been accepted. This solver
 * has shown empirically greater globalization strategy. Additionally, the
 * expected improvement computation considers the gaps in the dynamics:
 * \f{equation}
 *   \Delta J(\alpha) = \Delta_1\alpha + \frac{1}{2}\Delta_2\alpha^2,
 * \f}
 * with
 * \f{eqnarray}
 *   \Delta_1 = \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k}
 * +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k} -
 *   V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
 *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k +
 * \mathbf{\bar{f}}_k^\top(2 V_{\mathbf{xx}_k}\mathbf{x}_k
 * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). \f}
 *
 * For more details about the feasibility-driven differential dynamic
 * programming algorithm see:
 * \include mastalli-icra20.bib
 *
 * \sa `SolverDDP()`, `backwardPass()`, `forwardPass()`, `expectedImprovement()`
 * and `updateExpectedImprovement()`
 */
class SolverCSQP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the fadmm solver
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
   * @copybrief SolverAbstract::expectedImprovement
   *
   * This function requires to first run `updateExpectedImprovement()`. The
   * expected improvement computation considers the gaps in the dynamics:
   * \f{equation} \Delta J(\alpha) = \Delta_1\alpha +
   * \frac{1}{2}\Delta_2\alpha^2, \f} with
   * \f{eqnarray}
   *   \Delta_1 = \sum_{k=0}^{N-1} \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{u}_k}
   * +\mathbf{\bar{f}}_k^\top(V_{\mathbf{x}_k}
   * - V_{\mathbf{xx}_k}\mathbf{x}_k),\nonumber\\ \Delta_2 = \sum_{k=0}^{N-1}
   *   \mathbf{k}_k^\top\mathbf{Q}_{\mathbf{uu}_k}\mathbf{k}_k +
   * \mathbf{\bar{f}}_k^\top(2 V_{\mathbf{xx}_k}\mathbf{x}_k
   * - V_{\mathbf{xx}_k}\mathbf{\bar{f}}_k). \f}
   */
  // virtual const Eigen::Vector2d& expectedImprovement();

  /**
   * @brief Update internal values for computing the expected improvement
   */
  void updateExpectedImprovement();

  virtual void forwardPass(const double stepLength = 0.);
  virtual void forwardPass_without_constraints();
  virtual void backwardPass();
  virtual void backwardPass_without_rho_update();
  virtual void backwardPass_without_constraints();
  virtual void backwardPass_mt();
  virtual void backwardPass_without_rho_update_mt();

  /**
   * @brief Computes the merit function, gaps at the given xs, us along with
   * delta x and delta u
   */
  virtual void computeDirection(const bool recalcDiff);

  virtual double tryStep(const double stepLength);

  virtual void calc(const bool recalc = true);

  virtual void reset_params();

  virtual void reset_rho_vec();

  // virtual void set_constraints(const
  // std::vector<std::shared_ptr<ConstraintModelAbstract>>& constraint_models){
  //   constraint_models_ = constraint_models;
  // };

  /**
   * @brief Compute the KKT conditions residual
   */
  virtual void checkKKTConditions();

  const std::vector<Eigen::VectorXd>& get_xs_try() const { return xs_try_; };
  const std::vector<Eigen::VectorXd>& get_us_try() const { return us_try_; };

  const std::vector<Eigen::VectorXd>& get_xs() const { return xs_; };
  const std::vector<Eigen::VectorXd>& get_us() const { return us_; };

  const std::vector<Eigen::VectorXd>& get_dx_tilde() const { return dxtilde_; };
  const std::vector<Eigen::VectorXd>& get_du_tilde() const { return dutilde_; };

  const std::vector<Eigen::VectorXd>& get_dx() const { return dx_; };
  const std::vector<Eigen::VectorXd>& get_du() const { return du_; };

  const std::vector<Eigen::VectorXd>& get_y() const { return y_; };
  const std::vector<Eigen::VectorXd>& get_z() const { return z_; };

  const std::vector<Eigen::VectorXd>& get_rho_vec() const { return rho_vec_; };

  //   const std::vector<std::shared_ptr<ConstraintModelAbstract>>&
  //   get_constraints() const { return cmodels_; };

  double get_KKT() const { return KKT_; };
  double get_gap_norm() const { return gap_norm_; };
  double get_constraint_norm() const { return constraint_norm_; };
  double get_qp_iters() const { return qp_iters_; };
  double get_xgrad_norm() const { return x_grad_norm_; };
  double get_ugrad_norm() const { return u_grad_norm_; };
  double get_merit() const { return merit_; };
  bool get_extra_iteration_for_last_kkt() const {
    return extra_iteration_for_last_kkt_;
  };
  bool get_use_filter_line_search() const { return use_filter_line_search_; };
  double get_lag_mul_inf_norm_coef() const { return lag_mul_inf_norm_coef_; };
  double get_mu_dynamic() const { return mu_dynamic_; };
  double get_mu_constraint() const { return mu_constraint_; };
  double get_termination_tolerance() const { return termination_tol_; };
  std::size_t get_max_qp_iters() const { return max_qp_iters_; };
  bool get_equality_qp_initial_guess() const {
    return equality_qp_initial_guess_;
  };
  std::size_t get_filter_size() const { return filter_size_; };

  std::size_t get_rho_update_interval() const { return rho_update_interval_; };
  std::size_t get_adaptive_rho_tolerance() const {
    return adaptive_rho_tolerance_;
  };
  double get_alpha() const { return alpha_; };
  double get_sigma() const { return sigma_; };
  double get_rho_sparse() const { return rho_sparse_; };

  double get_eps_abs() const { return eps_abs_; };
  double get_eps_rel() const { return eps_rel_; };
  double get_norm_primal() const { return norm_primal_; };
  double get_norm_primal_tolerance() const { return norm_primal_tolerance_; };
  double get_norm_dual() const { return norm_dual_; };
  double get_norm_dual_tolerance() const { return norm_dual_tolerance_; };

  double get_reset_y() const { return reset_y_; };
  double get_reset_rho() const { return reset_rho_; };
  double get_rho_min() const { return rho_min_; };
  double get_rho_max() const { return rho_max_; };
  double get_max_solve_time() const { return max_solve_time_; };
  bool get_max_solve_time_reached() const { return max_solve_time_reached_; };

  bool getQPCallbacks() const { return with_qp_callbacks_; };

  void printQPCallbacks(const int iter);

  void setQPCallbacks(const bool inCallbacks);

  void set_rho_update_interval(const std::size_t interval) {
    rho_update_interval_ = interval;
  };
  void set_adaptive_rho_tolerance(const std::size_t tolerance) {
    adaptive_rho_tolerance_ = tolerance;
  };

  void set_lag_mul_inf_norm_coef(const double lag_mul_inf_norm_coef) {
    lag_mul_inf_norm_coef_ = lag_mul_inf_norm_coef;
  };
  void set_mu_dynamic(const double mu_dynamic) { mu_dynamic_ = mu_dynamic; };
  void set_mu_constraint(const double mu_constraint) {
    mu_constraint_ = mu_constraint;
  };
  void set_alpha(const double alpha) { alpha_ = alpha; };
  void set_sigma(const double sigma) { sigma_ = sigma; };

  void set_equality_qp_initial_guess(const bool equality_qp_initial_guess) {
    equality_qp_initial_guess_ = equality_qp_initial_guess;
  };

  void set_termination_tolerance(const double tol) { termination_tol_ = tol; };
  void set_extra_iteration_for_last_kkt(const bool inBool) {
    extra_iteration_for_last_kkt_ = inBool;
  };
  void set_use_filter_line_search(const bool inBool) {
    use_filter_line_search_ = inBool;
  };
  void set_filter_size(const std::size_t inFilterSize) {
    filter_size_ = inFilterSize;
    gap_list_.resize(filter_size_);
    constraint_list_.resize(filter_size_);
    cost_list_.resize(filter_size_);
  };

  void update_lagrangian_parameters(const int iter);
  void set_rho_sparse(const double rho_sparse) { rho_sparse_ = rho_sparse; };
  void update_rho_vec(const int iter);
  void apply_rho_update(const double rho_sparse);

  void set_max_qp_iters(const int iters) { max_qp_iters_ = iters; };
  void set_eps_abs(const double eps_abs) { eps_abs_ = eps_abs; };
  void set_eps_rel(const double eps_rel) { eps_rel_ = eps_rel; };
  void set_max_solve_time(const double max_solve_time) {
    max_solve_time_ = max_solve_time;
  };

 public:
  boost::circular_buffer<double>
      constraint_list_;  //!< memory buffer of constraint norms (used in filter
                         //!< line-search)
  boost::circular_buffer<double>
      gap_list_;  //!< memory buffer of gap norms (used in filter line-search)
  boost::circular_buffer<double>
      cost_list_;  //!< memory buffer of gap norms (used in filter line-search)
  using SolverDDP::cost_try_;
  using SolverDDP::us_try_;
  using SolverDDP::xs_try_;
  std::vector<Eigen::VectorXd>
      fs_try_;                       //!< Gaps/defects between shooting nodes
  std::vector<Eigen::VectorXd> dx_;  //!< the descent direction for x
  std::vector<Eigen::VectorXd> du_;  //!< the descent direction for u
  std::vector<Eigen::VectorXd>
      lag_mul_;  //!< the Lagrange multiplier of the dynamics constraint
  double lag_mul_inf_norm_;  //!< the infinite norm of Lagrange multiplier
  double lag_mul_inf_norm_coef_ =
      10.;  //!< merit function coefficient scaling the infinite norm of
            //!< Lagrange multiplier
  Eigen::VectorXd fs_flat_;  //!< Gaps/defects between shooting nodes (1D array)
  bool use_filter_line_search_ = true;  //!< Use filter line search

  std::vector<Eigen::VectorXd> dxtilde_;  //!< the descent direction for x
  std::vector<Eigen::VectorXd> dutilde_;  //!< the descent direction for u

  // ADMM parameters
  std::vector<Eigen::VectorXd> y_;            //!< lagrangian dual variable
  std::vector<Eigen::VectorXd> z_;            //!< second admm variable
  std::vector<Eigen::VectorXd> z_prev_;       //!< second admm variable previous
  std::vector<Eigen::VectorXd> z_relaxed_;    //!< relaxed step of z
  std::vector<Eigen::VectorXd> rho_vec_;      //!< rho vector
  std::vector<Eigen::VectorXd> inv_rho_vec_;  //!< rho vector

  double norm_primal_ = 0.0;      //!< norm primal residual
  double norm_dual_ = 0.0;        //!< norm dual residual
  double norm_primal_rel_ = 0.0;  //!< norm primal relative residual
  double norm_dual_rel_ = 0.0;    //!< norm dual relative residual
  double norm_primal_tolerance_ =
      0.0;                            //!< tolerance of the primal residual norm
  double norm_dual_tolerance_ = 0.0;  //!< tolerance of the primal residual norm
  bool reset_y_ = false;
  bool reset_rho_ = false;
  bool update_rho_with_heuristic_ = false;
  bool remove_reg_ = false;  //!< Removes Crocoddyl's regularization (preg,dreg)

 protected:
  double merit_ = 0;            //!< merit function at nominal traj
  double merit_try_ = 0;        //!< merit function for the step length tried
  double x_grad_norm_ = 0;      //!< 1 norm of the delta x
  double u_grad_norm_ = 0;      //!< 1 norm of the delta u
  double gap_norm_ = 0;         //!< 1 norm of the gaps
  double constraint_norm_ = 0;  //!< 1 norm of constraint violation
  double constraint_norm_try_ = 0;  //!< 1 norm of constraint violation try
  double gap_norm_try_ = 0;         //!< 1 norm of the gaps
  double mu_dynamic_ =
      1e1;  //!< penalty weight for dymanic violation in the merit function
  double mu_constraint_ =
      1e1;  //!< penalty weight for constraint violation in the merit function
  double termination_tol_ = 1e-6;  //!< Termination tolerance
  // bool with_callbacks_ = false;                                //!< With
  // callbacks
  bool with_qp_callbacks_ = false;  //!< With QP callbacks
  bool extra_iteration_for_last_kkt_ =
      false;             //!< Additional iteration if SQP max. iter reached
  double sigma_ = 1e-6;  //!< proximal term
  double alpha_ = 1.6;   //!< relaxed step size
  std::size_t max_qp_iters_ = 1000;  //!< max qp iters
  std::size_t qp_iters_ = 0;

  double rho_estimate_sparse_ = 0.0;  //!< rho estimate
  double rho_sparse_;                 //!< rho
  double rho_sparse_base_ = 1e-1;
  double rho_min_ = 1e-6;                 //!< rho min
  double rho_max_ = 1e3;                  //!< rho max
  std::size_t rho_update_interval_ = 25;  //!< frequency of update of rho
  double adaptive_rho_tolerance_ = 5;
  double eps_abs_ = 1e-4;  //!< absolute termination criteria
  double eps_rel_ = 1e-4;  //!< relative termination criteria
  double equality_qp_initial_guess_ =
      true;  //!< warm-start the QP with unconstrained solution
  std::size_t filter_size_ =
      1;  //!< Filter size for line-search (do not change the default value !)
  double KKT_ =
      std::numeric_limits<double>::infinity();  //!< KKT conditions residual

 private:
  double th_acceptnegstep_;  //!< Threshold used for accepting step along ascent
                             //!< direction
  bool is_worse_than_memory_ =
      false;  //!< Boolean for filter line-search criteria

  Eigen::VectorXd tmp_vec_x_;                    //!< Temporary variable
  std::vector<Eigen::VectorXd> tmp_vec_u_;       //!< Temporary variable
  std::vector<Eigen::VectorXd> tmp_dual_cwise_;  //!< Temporary variable
  Eigen::VectorXd tmp_Vx_;                       //!< Temporary variable
  std::vector<Eigen::VectorXd> tmp_Cdx_Cdu_;     //!< Temporary variable
  std::vector<Eigen::MatrixXd> tmp_rhoGx_mat_;   //!< Temporary variable
  std::vector<Eigen::MatrixXd> tmp_rhoGu_mat_;   //!< Temporary variable
  std::vector<Eigen::VectorXd> Vxx_fs_;          //!< Temporary variable

  double start_time_ = 0.0;  // Time when the solve function was called
  bool max_solve_time_reached_ = false;  // Flag indicating solver timedout
  double max_solve_time_ =
      std::numeric_limits<double>::infinity();  // Maximum time in seconds used
                                                // to stop execution of the
                                                // solver
};

}  // namespace mim_solvers

#endif  // MIM_SOLVERS_CSQP_HPP_

// To-do: move definitions to a dedicated file

// Same logic as in Proxsuite and Pinocchio to check eigen malloc
#ifdef MIM_SOLVERS_EIGEN_CHECK_MALLOC
#ifndef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#define EIGEN_RUNTIME_NO_MALLOC
#endif
#endif

// #include <Eigen/Core>
// #include <cassert>

#ifdef MIM_SOLVERS_EIGEN_CHECK_MALLOC
#ifdef EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#undef EIGEN_RUNTIME_NO_MALLOC
#undef EIGEN_RUNTIME_NO_MALLOC_WAS_NOT_DEFINED
#endif
#endif

// Check memory allocation for Eigen
#ifdef MIM_SOLVERS_EIGEN_CHECK_MALLOC
#define MIM_SOLVERS_EIGEN_MALLOC(allowed) \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#define MIM_SOLVERS_EIGEN_MALLOC_ALLOWED() MIM_SOLVERS_EIGEN_MALLOC(true)
#define MIM_SOLVERS_EIGEN_MALLOC_NOT_ALLOWED() MIM_SOLVERS_EIGEN_MALLOC(false)
#else
#define MIM_SOLVERS_EIGEN_MALLOC(allowed)
#define MIM_SOLVERS_EIGEN_MALLOC_ALLOWED()
#define MIM_SOLVERS_EIGEN_MALLOC_NOT_ALLOWED()
#endif
