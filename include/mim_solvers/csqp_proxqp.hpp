///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_CSQP_PROXQP_HPP_
#define MIM_SOLVERS_CSQP_PROXQP_HPP_

#include <Eigen/Cholesky>
#include <boost/circular_buffer.hpp>
#include <proxsuite/proxqp/sparse/sparse.hpp>
#include <vector>

#include "mim_solvers/ddp.hpp"

namespace mim_solvers {

/**
 * @brief PROXQP solver
 *
 * The PROXQP solver computes an optimal trajectory and control commands by
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
class SolverPROXQP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the PROXQP solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverPROXQP(std::shared_ptr<crocoddyl::ShootingProblem> problem);
  virtual ~SolverPROXQP();

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

  /**
   * @brief Computes the merit function, gaps at the given xs, us along with
   * delta x and delta u
   */
  virtual void computeDirection(const bool recalcDiff);

  virtual double tryStep(const double stepLength);

  virtual void calc(const bool recalc = true);

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

  const std::vector<Eigen::VectorXd>& get_fs() const { return fs_; };

  const std::vector<Eigen::VectorXd>& get_dx() const { return dx_; };
  const std::vector<Eigen::VectorXd>& get_du() const { return du_; };

  const std::vector<Eigen::VectorXd>& get_y() const { return y_; };
  const std::vector<Eigen::VectorXd>& get_lag_mul() const { return lag_mul_; };

  double get_KKT() const { return KKT_; };
  double get_gap_norm() const { return gap_norm_; };
  double get_constraint_norm() const { return constraint_norm_; };
  double get_qp_iters() const { return qp_iters_; };
  double get_xgrad_norm() const { return x_grad_norm_; };
  double get_ugrad_norm() const { return u_grad_norm_; };
  double get_merit() const { return merit_; };
  bool get_use_filter_line_search() const { return use_filter_line_search_; };
  double get_mu_dynamic() const { return mu_dynamic_; };
  double get_mu_constraint() const { return mu_constraint_; };
  double get_termination_tolerance() const { return termination_tol_; };
  int get_max_qp_iters() { return max_qp_iters_; };
  double get_cost() { return cost_; };
  std::size_t get_filter_size() const { return filter_size_; };

  double get_eps_abs() { return eps_abs_; };
  double get_eps_rel() { return eps_rel_; };
  double get_norm_primal() { return norm_primal_; };
  double get_norm_dual() { return norm_dual_; };

  void printCallbacks();
  void setCallbacks(bool inCallbacks);
  bool getCallbacks();

  void set_mu_dynamic(double mu_dynamic) { mu_dynamic_ = mu_dynamic; };
  void set_mu_constraint(double mu_constraint) {
    mu_constraint_ = mu_constraint;
  };

  void set_termination_tolerance(double tol) { termination_tol_ = tol; };
  void set_use_filter_line_search(bool inBool) {
    use_filter_line_search_ = inBool;
  };
  void set_filter_size(const std::size_t inFilterSize) {
    filter_size_ = inFilterSize;
    gap_list_.resize(filter_size_);
    constraint_list_.resize(filter_size_);
    cost_list_.resize(filter_size_);
  };

  void set_max_qp_iters(int iters) { max_qp_iters_ = iters; };

  const Eigen::MatrixXd& get_P() const { return P_; };
  const Eigen::MatrixXd& get_A() const { return A_; };
  const Eigen::MatrixXd& get_C() const { return C_; };
  const Eigen::VectorXd& get_q() const { return q_; };
  const Eigen::VectorXd& get_b() const { return b_; };
  const Eigen::VectorXd& get_l() const { return l_; };
  const Eigen::VectorXd& get_u() const { return u_; };

  void set_eps_abs(double eps_abs) { eps_abs_ = eps_abs; };
  void set_eps_rel(double eps_rel) { eps_rel_ = eps_rel; };

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
  std::vector<Eigen::VectorXd> y_;  //!< lagrangian dual variable

  Eigen::VectorXd fs_flat_;  //!< Gaps/defects between shooting nodes (1D array)
  double KKT_ =
      std::numeric_limits<double>::infinity();  //!< KKT conditions residual
  bool use_filter_line_search_ = true;          //!< Use filter line search

 protected:
  double merit_ = 0;            //!< merit function at nominal traj
  double merit_try_ = 0;        //!< merit function for the step length tried
  double x_grad_norm_ = 0;      //!< 1 norm of the delta x
  double u_grad_norm_ = 0;      //!< 1 norm of the delta u
  double gap_norm_ = 0;         //!< 1 norm of the gaps
  double constraint_norm_ = 0;  //!< 1 norm of constraint violation
  double constraint_norm_try_ = 0;  //!< 1 norm of constraint violation try
  double gap_norm_try_ = 0;         //!< 1 norm of the gaps
  double cost_ = 0.0;               //!< cost function
  double mu_dynamic_ = 1e1;         //!< penalty no constraint violation
  double mu_constraint_ = 1e1;      //!< penalty no constraint violation
  double termination_tol_ = 1e-8;   //!< Termination tolerance
  bool with_callbacks_ = false;     //!< With callbacks
  int max_qp_iters_ = 1000;         //!< maximum number of QP iterations
  int qp_iters_ = 0;                //!< current number of QP iterations

  double eps_abs_ = 1e-4;  //!< absolute termination criteria
  double eps_rel_ = 1e-4;  //!< relative termination criteria
  std::size_t filter_size_ =
      1;  //!< Filter size for line-search (do not change the default value !)

  double norm_primal_ = 0.0;  //!< norm primal residual
  double norm_dual_ = 0.0;    //!< norm dual residual

  // PROX QP STUFF
  Eigen::MatrixXd P_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd C_;

  Eigen::SparseMatrix<double> Psp_;
  Eigen::SparseMatrix<double> Asp_;
  Eigen::SparseMatrix<double> Csp_;

  Eigen::VectorXd q_;
  Eigen::VectorXd b_;
  Eigen::VectorXd l_;
  Eigen::VectorXd u_;

  int n_in = 0;
  int n_eq = 0;
  int n_vars = 0;
  //
 private:
  double th_acceptnegstep_;  //!< Threshold used for accepting step along ascent
                             //!< direction
  Eigen::VectorXd dual_vecx;
  Eigen::VectorXd dual_vecu;
  bool is_worse_than_memory_ =
      false;  //!< Boolean for filter line-search criteria
};

}  // namespace mim_solvers

#endif  // MIM_SOLVERS_CSQP_PROXQP_HPP_
