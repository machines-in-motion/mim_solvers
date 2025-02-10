///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_SQP_HPP_
#define MIM_SOLVERS_SQP_HPP_

#include <Eigen/Cholesky>
#include <boost/circular_buffer.hpp>
#include <vector>

#include "mim_solvers/ddp.hpp"
#include "mim_solvers/utils/callbacks.hpp"

namespace mim_solvers {

/**
 * @brief SQP solver
 *
 * The SQP solver computes an optimal trajectory and control commands by
 * iterates running `backwardPass()` and `forwardPass()`. The backward pass
 * accepts infeasible guess as described in the
 * `crocoddyl::SolverDDP::backwardPass()`. Additionally, the forward pass
 * handles infeasibility simulations that resembles the numerical behaviour of
 * a multiple-shooting formulation, i.e.:
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
 * \sa `crocoddyl::SolverDDP()`, `backwardPass()`, `forwardPass()`,
 * `expectedImprovement()` and `updateExpectedImprovement()`
 */
class SolverSQP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the SQP solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverSQP(std::shared_ptr<crocoddyl::ShootingProblem> problem);
  virtual ~SolverSQP();

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

  virtual void forwardPass(const double stepLength = 0);
  /**
   * @brief Computes the merit function, gaps at the given xs, us along with
   * delta x and delta u
   */
  virtual void computeDirection(const bool recalcDiff);

  virtual double tryStep(const double stepLength);

  /**
   * @brief Compute the KKT conditions residual
   */
  virtual void checkKKTConditions();

  const std::vector<Eigen::VectorXd>& get_xs_try() const { return xs_try_; };
  const std::vector<Eigen::VectorXd>& get_us_try() const { return us_try_; };

  const std::vector<Eigen::VectorXd>& get_dx() const { return dx_; };
  const std::vector<Eigen::VectorXd>& get_du() const { return du_; };

  double get_KKT() const { return KKT_; };
  double get_gap_norm() const { return gap_norm_; };
  double get_xgrad_norm() const { return x_grad_norm_; };
  double get_ugrad_norm() const { return u_grad_norm_; };
  double get_merit() const { return merit_; };
  bool get_extra_iteration_for_last_kkt() const {
    return extra_iteration_for_last_kkt_;
  };
  bool get_use_filter_line_search() const { return use_filter_line_search_; };
  double get_mu_dynamic() const { return mu_dynamic_; };
  double get_termination_tolerance() const { return termination_tol_; };
  std::size_t get_filter_size() const { return filter_size_; };

  void set_mu_dynamic(double mu_dynamic) { mu_dynamic_ = mu_dynamic; };
  void set_termination_tolerance(double tol) { termination_tol_ = tol; };
  void set_extra_iteration_for_last_kkt(bool inBool) {
    extra_iteration_for_last_kkt_ = inBool;
  };
  void set_use_filter_line_search(bool inBool) {
    use_filter_line_search_ = inBool;
  };
  void set_filter_size(const std::size_t inFilterSize) {
    filter_size_ = inFilterSize;
    gap_list_.resize(filter_size_);
    cost_list_.resize(filter_size_);
  };

 public:
  using SolverDDP::cost_try_;
  using SolverDDP::us_try_;
  using SolverDDP::xs_try_;
  std::vector<Eigen::VectorXd>
      fs_try_;                       //!< Gaps/defects between shooting nodes
  std::vector<Eigen::VectorXd> dx_;  //!< the descent direction for x
  std::vector<Eigen::VectorXd> du_;  //!< the descent direction for u
  std::vector<Eigen::VectorXd>
      lag_mul_;  //!< the Lagrange multiplier of the dynamics constraint
  boost::circular_buffer<double>
      gap_list_;  //!< memory buffer of gap norms (used in filter line-search)
  boost::circular_buffer<double>
      cost_list_;  //!< memory buffer of gap norms (used in filter line-search)
  Eigen::VectorXd fs_flat_;  //!< Gaps/defects between shooting nodes (1D array)
  bool use_filter_line_search_ = true;  //!< Use filter line search

 protected:
  double merit_ = 0;         //!< merit function at nominal traj
  double merit_try_ = 0;     //!< merit function for the step length tried
  double x_grad_norm_ = 0;   //!< 1 norm of the delta x
  double u_grad_norm_ = 0;   //!< 1 norm of the delta u
  double gap_norm_ = 0;      //!< 1 norm of the gaps
  double gap_norm_try_ = 0;  //!< 1 norm of the gaps
  double mu_dynamic_ =
      1e0;  //!< penalty weight for dymanic violation in the merit function
  double termination_tol_ = 1e-6;  //!< Termination tolerance
  // bool with_callbacks_ = false;                                //!< With
  // callbacks
  bool extra_iteration_for_last_kkt_ =
      false;  //!< Additional iteration if SQP max. iter reached
  std::size_t filter_size_ =
      1;  //!< Filter size for line-search (do not change the default value !)
  double KKT_ =
      std::numeric_limits<double>::infinity();  //!< KKT conditions residual

 private:
  double th_acceptnegstep_;  //!< Threshold used for accepting step along ascent
                             //!< direction
  bool is_worse_than_memory_ =
      false;                   //!< Boolean for filter line-search criteria
  Eigen::VectorXd tmp_vec_x_;  //!< Temporary variable
  std::vector<Eigen::VectorXd> tmp_vec_u_;  //!< Temporary variable
};

}  // namespace mim_solvers

#endif  // MIM_SOLVERS_SQP_HPP_
