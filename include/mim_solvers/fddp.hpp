///////////////////////////////////////////////////////////////////////////////
//
// This file is a modified version of SolverDDP from the Crocoddyl library
// This modified version is used for benchmarking purposes only
// Original file :
// https://github.com/loco-3d/crocoddyl/blob/devel/include/crocoddyl/core/solvers/fddp.hpp
//
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_FDDP_HPP_
#define MIM_SOLVERS_FDDP_HPP_

#include <Eigen/Cholesky>
#include <boost/circular_buffer.hpp>
#include <vector>

#include "mim_solvers/ddp.hpp"

namespace mim_solvers {

/**
 * @brief Feasibility-driven Differential Dynamic Programming (FDDP) solver
 *
 * The FDDP solver computes an optimal trajectory and control commands by
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
class SolverFDDP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the FDDP solver
   *
   * @param[in] problem  shooting problem
   */
  explicit SolverFDDP(std::shared_ptr<crocoddyl::ShootingProblem> problem);
  virtual ~SolverFDDP();

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
  virtual const Eigen::Vector2d& expectedImprovement();

  /**
   * @brief Update internal values for computing the expected improvement
   */
  void updateExpectedImprovement();
  virtual void forwardPass(const double stepLength);

  /**
   * @brief Computes the merit function, gaps at the given xs, us along with
   * delta x and delta u
   */
  virtual void computeDirection(const bool recalcDiff);

  /**
   * @brief Return the threshold used for accepting step along ascent direction
   */
  double get_th_acceptnegstep() const;

  /**
   * @brief Modify the threshold used for accepting step along ascent direction
   */
  void set_th_acceptnegstep(const double th_acceptnegstep);

  /**
   * @brief Compute the KKT conditions residual
   */
  virtual void checkKKTConditions();

  void set_termination_tolerance(double tol) { termination_tol_ = tol; };

  double get_termination_tolerance() const { return termination_tol_; };

  bool get_use_filter_line_search() const { return use_filter_line_search_; };
  std::size_t get_filter_size() const { return filter_size_; };
  void set_use_filter_line_search(bool inBool) {
    use_filter_line_search_ = inBool;
  };
  void set_filter_size(const std::size_t inFilterSize) {
    filter_size_ = inFilterSize;
    gap_list_.resize(filter_size_);
    cost_list_.resize(filter_size_);
  };
  double get_gap_norm() const { return gap_norm_; };
  double get_KKT() const { return KKT_; };

  boost::circular_buffer<double>
      gap_list_;  //!< memory buffer of gap norms (used in filter line-search)
  boost::circular_buffer<double>
      cost_list_;  //!< memory buffer of gap norms (used in filter line-search)
  bool use_filter_line_search_ = true;  //!< Use filter line search
  std::size_t filter_size_ =
      1;  //!< Filter size for line-search (do not change the default value !)
  bool is_worse_than_memory_ =
      false;  //!< Boolean for filter line-search criteria

  std::vector<Eigen::VectorXd>
      lag_mul_;  //!< the Lagrange multiplier of the dynamics constraint
  Eigen::VectorXd fs_flat_;  //!< Gaps/defects between shooting nodes (1D array)
  double termination_tol_ = 1e-6;  //!< Termination tolerance

  std::vector<Eigen::VectorXd>
      fs_try_;  //!< Gaps/defects between shooting nodes

  double gap_norm_ = 0;      //!< 1 norm of the gaps
  double gap_norm_try_ = 0;  //!< 1 norm of the gaps

 protected:
  double dg_;  //!< Internal data for computing the expected improvement
  double dq_;  //!< Internal data for computing the expected improvement
  double dv_;  //!< Internal data for computing the expected improvement
  double KKT_ =
      std::numeric_limits<double>::infinity();  //!< KKT conditions residual

 private:
  double th_acceptnegstep_;  //!< Threshold used for accepting step along ascent
                             //!< direction
};

}  // namespace mim_solvers

#endif  // MIM_SOLVERSS_FDDP_HPP_
