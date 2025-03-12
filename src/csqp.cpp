///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif  // CROCODDYL_WITH_MULTITHREADING

#include <crocoddyl/core/solver-base.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <iomanip>
#include <iostream>

#include "mim_solvers/csqp.hpp"

using namespace crocoddyl;

namespace mim_solvers {

SolverCSQP::SolverCSQP(std::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverDDP(problem) {
  const std::size_t T = this->problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  constraint_list_.resize(filter_size_);
  gap_list_.resize(filter_size_);
  cost_list_.resize(filter_size_);

  fs_flat_.resize(ndx * (T + 1));
  fs_flat_.setZero();

  xs_try_.resize(T + 1);
  us_try_.resize(T);
  dx_.resize(T + 1);
  du_.resize(T);
  dxtilde_.resize(T + 1);
  dutilde_.resize(T);
  lag_mul_.resize(T + 1);
  fs_try_.resize(T + 1);

  z_.resize(T + 1);
  z_relaxed_.resize(T + 1);
  z_prev_.resize(T + 1);
  y_.resize(T + 1);
  rho_vec_.resize(T + 1);
  inv_rho_vec_.resize(T + 1);
  rho_sparse_ = rho_sparse_base_;
  std::size_t n_eq_crocoddyl = 0;

  tmp_Vx_.resize(ndx);
  tmp_Vx_.setZero();
  tmp_vec_x_.resize(ndx);
  tmp_vec_x_.setZero();

  tmp_Cdx_Cdu_.resize(T + 1);
  tmp_dual_cwise_.resize(T + 1);
  tmp_rhoGx_mat_.resize(T + 1);
  tmp_rhoGu_mat_.resize(T);
  tmp_vec_u_.resize(T);
  Vxx_fs_.resize(T);

  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    std::size_t nc = model->get_ng();
    n_eq_crocoddyl += model->get_nh();

    xs_try_[t] = model->get_state()->zero();
    us_try_[t] = Eigen::VectorXd::Zero(nu);
    dx_[t].resize(ndx);
    dx_[t].setZero();
    du_[t].resize(nu);
    du_[t] = Eigen::VectorXd::Zero(nu);
    dxtilde_[t].resize(ndx);
    dxtilde_[t].setZero();
    dutilde_[t].resize(nu);
    dutilde_[t] = Eigen::VectorXd::Zero(nu);
    lag_mul_[t].resize(ndx);
    lag_mul_[t].setZero();
    fs_try_[t].resize(ndx);
    fs_try_[t] = Eigen::VectorXd::Zero(ndx);

    z_[t].resize(nc);
    z_[t].setZero();
    z_relaxed_[t].resize(nc);
    z_relaxed_[t].setZero();
    z_prev_[t].resize(nc);
    z_prev_[t].setZero();
    y_[t].resize(nc);
    y_[t].setZero();

    tmp_Cdx_Cdu_[t].resize(nc);
    tmp_Cdx_Cdu_[t].setZero();
    tmp_dual_cwise_[t].resize(nc);
    tmp_dual_cwise_[t].setZero();
    tmp_rhoGx_mat_[t].resize(nc, ndx);
    tmp_rhoGx_mat_[t].setZero();
    tmp_rhoGu_mat_[t].resize(nc, nu);
    tmp_rhoGu_mat_[t].setZero();
    tmp_vec_u_[t].resize(nu);
    tmp_vec_u_[t].setZero();
    Vxx_fs_[t].resize(ndx);
    Vxx_fs_[t].setZero();

    rho_vec_[t].resize(nc);
    inv_rho_vec_[t].resize(nc);
  }

  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();
  dx_.back().resize(ndx);
  dx_.back().setZero();
  dxtilde_.back().resize(ndx);
  dxtilde_.back().setZero();
  lag_mul_.back().resize(ndx);
  lag_mul_.back().setZero();
  fs_try_.back().resize(ndx);
  fs_try_.back() = Eigen::VectorXd::Zero(ndx);

  const std::size_t nc = problem_->get_terminalModel()->get_ng();

  z_.back().resize(nc);
  z_.back().setZero();
  z_relaxed_.back().resize(nc);
  z_relaxed_.back().setZero();
  z_prev_.back().resize(nc);
  z_prev_.back().setZero();
  y_.back().resize(nc);
  y_.back().setZero();

  tmp_Cdx_Cdu_.back().resize(nc);
  tmp_Cdx_Cdu_.back().setZero();
  tmp_dual_cwise_.back().resize(nc);
  tmp_dual_cwise_.back().setZero();
  tmp_rhoGx_mat_.back().resize(nc, ndx);
  tmp_rhoGx_mat_.back().setZero();

  rho_vec_.back().resize(nc);
  inv_rho_vec_.back().resize(nc);

  // Check that no equality constraint was specified through Crocoddyl's API
  n_eq_crocoddyl += problem_->get_terminalModel()->get_nh();
  if (n_eq_crocoddyl != 0) {
    throw_pretty(
        "Error: nh must be zero !!! Crocoddyl's equality constraints API is "
        "not supported by mim_solvers.\n"
        "  >> Equality constraints of the form H(x,u) = h must be implemented "
        "as g <= G(x,u) <= g by specifying \n"
        "     lower and upper bounds in the constructor of the constraint "
        "model residual or by setting g_ub and g_lb.")
  }

  const std::size_t n_alphas = 10;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
  if (th_stepinc_ < alphas_[n_alphas - 1]) {
    th_stepinc_ = alphas_[n_alphas - 1];
    std::cerr << "Warning: th_stepinc has higher value than lowest alpha "
                 "value, set to "
              << std::to_string(alphas_[n_alphas - 1]) << std::endl;
  }
}

void SolverCSQP::reset_params() {
  if (reset_rho_) {
    reset_rho_vec();
  }

  const std::size_t T = this->problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    z_[t].setZero();
    z_prev_[t].setZero();
    z_relaxed_[t].setZero();

    if (reset_y_) {
      y_[t].setZero();
    }
  }

  z_.back().setZero();
  z_prev_.back().setZero();
  z_relaxed_.back().setZero();

  if (reset_y_) {
    y_.back().setZero();
  }
}

SolverCSQP::~SolverCSQP() {}

bool SolverCSQP::solve(const std::vector<Eigen::VectorXd>& init_xs,
                       const std::vector<Eigen::VectorXd>& init_us,
                       const std::size_t maxiter, const bool /*is_feasible*/,
                       const double reginit) {
  START_PROFILER("SolverCSQP::solve");

  start_time_ = crocoddyl::getProfiler().take_time();

  if (problem_->is_updated()) {
    resizeData();
  }
  setCandidate(init_xs, init_us, false);
  // Otherwise xs[0] is overwritten by init_xs inside setCandidate()
  xs_[0] = problem_->get_x0();

  // it is needed in case that init_xs[0] is infeasible
  xs_try_[0] = problem_->get_x0();

  // Optionally remove Crocoddyl's regularization
  if (remove_reg_) {
    preg_ = 0.;
    dreg_ = 0.;
  } else {
    if (std::isnan(reginit)) {
      preg_ = reg_min_;
      dreg_ = reg_min_;
    } else {
      preg_ = reginit;
      dreg_ = reginit;
    }
  }

  // Otherwise benchmarks blowup
  // TODO: find cleaner way
  if (maxiter == 0) {
    calc(true);
    reset_rho_vec();
  }

  // Main SQP loop
  max_solve_time_reached_ = false;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    if (crocoddyl::getProfiler().take_time() - start_time_ >= max_solve_time_) {
      max_solve_time_reached_ = true;
      break;
    }
    // Compute gradients
    calc(true);

    // reset rho only at the beginning of each solve if reset_rho_ is false
    // (after calc to get correct lb and ub)
    if (iter_ == 0 && !reset_rho_) {
      reset_rho_vec();
    }

    // Solve QP
    if (remove_reg_) {
      computeDirection(true);
    } else {
      while (!max_solve_time_reached_) {
        try {
          computeDirection(true);
        } catch (std::exception& e) {
          increaseRegularization();
          if (preg_ >= reg_max_) {
            STOP_PROFILER("SolverCSQP::solve");
            return false;
          } else {
            continue;
          }
        }
        break;
      }
    }
    if (qp_iters_ == 0) {
      STOP_PROFILER("SolverCSQP::solve");
      return false;
    }

    // Check KKT criteria
    checkKKTConditions();

    // Perform callbacks
    for (const auto& callback : callbacks_) {
      (*callback)(*this, "CSQP");
    }

    if (KKT_ <= termination_tol_) {
      STOP_PROFILER("SolverCSQP::solve");
      return true;
    }

    // Line search
    constraint_list_.push_back(constraint_norm_);
    gap_list_.push_back(gap_norm_);
    cost_list_.push_back(cost_);

    // Calculate the coefficient of the merit function.
    if (mu_dynamic_ < 0. || mu_constraint_ < 0.) {
      lag_mul_inf_norm_ = 0;
      for (const auto& lag_mul : lag_mul_) {
        lag_mul_inf_norm_ =
            std::max(lag_mul_inf_norm_, lag_mul.lpNorm<Eigen::Infinity>());
      }
      for (const auto& y : y_) {
        lag_mul_inf_norm_ =
            std::max(lag_mul_inf_norm_, y.lpNorm<Eigen::Infinity>());
      }
      merit_ = cost_ + lag_mul_inf_norm_coef_ * lag_mul_inf_norm_ *
                           (gap_norm_ + constraint_norm_);
    } else {
      merit_ =
          cost_ + mu_dynamic_ * gap_norm_ + mu_constraint_ * constraint_norm_;
    }

    // We need to recalculate the derivatives when the step length passes
    bool found = false;
    // less than filter_size_, less or equal iter_
    const std::size_t max_count = std::min(filter_size_, iter_ + 1);
    for (const double steplength_ : alphas_) {
      try {
        merit_try_ = tryStep(steplength_);
      } catch (std::exception& e) {
        continue;
      }
      // Filter line search criteria
      if (use_filter_line_search_) {
        is_worse_than_memory_ = false;
        std::size_t count = 0.;
        while (count < max_count && !is_worse_than_memory_) {
          is_worse_than_memory_ =
              cost_list_[filter_size_ - 1 - count] <= cost_try_ &&
              gap_list_[filter_size_ - 1 - count] <= gap_norm_try_ &&
              constraint_list_[filter_size_ - 1 - count] <=
                  constraint_norm_try_;
          count++;
        }
        if (!is_worse_than_memory_) {
          setCandidate(xs_try_, us_try_, false);
          found = true;
          break;
        }
      }
      // Line-search criteria using merit function
      else {
        if (merit_ > merit_try_) {
          setCandidate(xs_try_, us_try_, false);
          found = true;
          break;
        }
      }
    }
    if (!found) {
      break;
    }

    // Regularization logic
    if (!remove_reg_) {
      if (steplength_ > th_stepdec_) {
        decreaseRegularization();
      } else {
        increaseRegularization();
        // preg_ equal to reg_max_
        if (preg_ >= reg_max_) {
          STOP_PROFILER("SolverCSQP::solve");
          return false;
        }
      }
    }
  }

  // If reached max iter and timeout not reached, still compute KKT residual
  if (extra_iteration_for_last_kkt_ && !max_solve_time_reached_) {
    // Compute gradients
    calc(true);

    // Solve QP
    if (remove_reg_) {
      computeDirection(true);
    } else {
      while (true) {
        try {
          computeDirection(true);
        } catch (std::exception& e) {
          increaseRegularization();
          // preg_ equal to reg_max_
          if (preg_ >= reg_max_) {
            return false;
          } else {
            continue;
          }
        }
        break;
      }
    }

    // Check KKT criteria
    checkKKTConditions();

    // Perform callbacks
    for (const auto& callback : callbacks_) {
      (*callback)(*this, "CSQP");
    }

    if (KKT_ <= termination_tol_) {
      STOP_PROFILER("SolverCSQP::solve");
      return true;
    }
  }

  STOP_PROFILER("SolverCSQP::solve");
  return false;
}

void SolverCSQP::calc(const bool recalc) {
  if (recalc) {
    problem_->calc(xs_, us_);
    cost_ = problem_->calcDiff(xs_, us_);
  }

  gap_norm_ = 0.;
  constraint_norm_ = 0.;
  // double infty = std::numeric_limits<double>::infinity();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    m->get_state()->diff(xs_[t + 1], d->xnext, fs_[t + 1]);

    gap_norm_ += fs_[t + 1].lpNorm<1>();

    const std::size_t nc = m->get_ng();
    constraint_norm_ +=
        (m->get_g_lb() - d->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
    constraint_norm_ +=
        (d->g - m->get_g_ub()).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
  }

  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();
  const std::size_t nc = problem_->get_terminalModel()->get_ng();

  constraint_norm_ += (problem_->get_terminalModel()->get_g_lb() - d_T->g)
                          .cwiseMax(Eigen::VectorXd::Zero(nc))
                          .lpNorm<1>();
  constraint_norm_ += (d_T->g - problem_->get_terminalModel()->get_g_ub())
                          .cwiseMax(Eigen::VectorXd::Zero(nc))
                          .lpNorm<1>();
}

void SolverCSQP::computeDirection(const bool /*recalcDiff*/) {
  // MIM_SOLVERS_EIGEN_MALLOC_NOT_ALLOWED();

  START_PROFILER("SolverCSQP::computeDirection");

  reset_params();

  if (equality_qp_initial_guess_) {
    backwardPass_without_constraints();
    forwardPass_without_constraints();
  }

  if (with_qp_callbacks_) {
    printQPCallbacks(0);
  }

  for (qp_iters_ = 1; qp_iters_ < max_qp_iters_ + 1; ++qp_iters_) {
    if (crocoddyl::getProfiler().take_time() - start_time_ >= max_solve_time_) {
      // Reduce number of QP iterations, to match real number of executed loops
      qp_iters_--;
      max_solve_time_reached_ = true;
      break;
    }
    if (qp_iters_ % rho_update_interval_ == 1 || rho_update_interval_ == 1) {
#ifdef CROCODDYL_WITH_MULTITHREADING
      if (problem_->get_nthreads() > 1)
        backwardPass_mt();
      else
#endif  // CROCODDYL_WITH_MULTITHREADING
        backwardPass();
    } else {
#ifdef CROCODDYL_WITH_MULTITHREADING
      if (problem_->get_nthreads() > 1)
        backwardPass_without_rho_update_mt();
      else
#endif  // CROCODDYL_WITH_MULTITHREADING
        backwardPass_without_rho_update();
    }
    forwardPass();
    update_lagrangian_parameters(qp_iters_);
    update_rho_vec(qp_iters_);

    // Because (eps_rel=0) x inf = NaN
    if (qp_iters_ % rho_update_interval_ == 0) {
      if (with_qp_callbacks_) {
        printQPCallbacks(qp_iters_);
      }
      if (std::fabs(eps_rel_) <= std::numeric_limits<double>::epsilon()) {
        norm_primal_tolerance_ = eps_abs_;
        norm_dual_tolerance_ = eps_abs_;
      } else {
        norm_primal_tolerance_ = eps_abs_ + eps_rel_ * norm_primal_rel_;
        norm_dual_tolerance_ = eps_abs_ + eps_rel_ * norm_dual_rel_;
      }
      if (norm_primal_ <= norm_primal_tolerance_ &&
          norm_dual_ <= norm_dual_tolerance_) {
        break;
      }
    }
  }

  STOP_PROFILER("SolverCSQP::computeDirection");
  // MIM_SOLVERS_EIGEN_MALLOC_ALLOWED();
}

void SolverCSQP::update_rho_vec(const int iter) {
  const double scale = std::sqrt((norm_primal_ * norm_dual_rel_) /
                                 (norm_dual_ * norm_primal_rel_));
  rho_estimate_sparse_ =
      std::min(std::max(scale * rho_sparse_, rho_min_), rho_max_);

  if (iter % rho_update_interval_ == 0) {  // && iter > 1){
    if (rho_estimate_sparse_ > rho_sparse_ * adaptive_rho_tolerance_ ||
        rho_estimate_sparse_ < rho_sparse_ / adaptive_rho_tolerance_) {
      rho_sparse_ = rho_estimate_sparse_;
      apply_rho_update(rho_sparse_);
    }
  }
}

void SolverCSQP::reset_rho_vec() {
  rho_sparse_ = rho_sparse_base_;
  apply_rho_update(rho_sparse_);
}

void SolverCSQP::apply_rho_update(const double rho_sparse_tmp) {
  START_PROFILER("SolverCSQP::apply_rho_update");
  const std::size_t T = this->problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  double infty = std::numeric_limits<double>::infinity();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::size_t nc = m->get_ng();

    for (std::size_t k = 0; k < nc; ++k) {
      if (m->get_g_lb()[k] == -infty && m->get_g_ub()[k] == infty) {
        rho_vec_[t][k] = rho_min_;
        inv_rho_vec_[t][k] = 1. / rho_min_;
      } else if (abs(m->get_g_lb()[k] - m->get_g_ub()[k]) <= 1e-6) {
        rho_vec_[t][k] = 1e3 * rho_sparse_tmp;
        inv_rho_vec_[t][k] = 1. / (1e3 * rho_sparse_tmp);
      } else if (m->get_g_lb()[k] < m->get_g_ub()[k]) {
        rho_vec_[t][k] = rho_sparse_tmp;
        inv_rho_vec_[t][k] = 1. / rho_sparse_tmp;
      }
    }
  }

  const std::size_t nc = problem_->get_terminalModel()->get_ng();

  for (std::size_t k = 0; k < nc; ++k) {
    if (problem_->get_terminalModel()->get_g_lb()[k] == -infty &&
        problem_->get_terminalModel()->get_g_ub()[k] == infty) {
      rho_vec_.back()[k] = rho_min_;
      inv_rho_vec_.back()[k] = 1. / rho_min_;
    } else if (abs(problem_->get_terminalModel()->get_g_lb()[k] -
                   problem_->get_terminalModel()->get_g_ub()[k]) <= 1e-6) {
      rho_vec_.back()[k] = 1e3 * rho_sparse_tmp;
      inv_rho_vec_.back()[k] = 1. / (1e3 * rho_sparse_tmp);
    } else if (problem_->get_terminalModel()->get_g_lb()[k] <
               problem_->get_terminalModel()->get_g_ub()[k]) {
      rho_vec_.back()[k] = rho_sparse_tmp;
      inv_rho_vec_.back()[k] = 1. / rho_sparse_tmp;
    }
  }
  STOP_PROFILER("SolverCSQP::apply_rho_update");
}

void SolverCSQP::checkKKTConditions() {
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();
  x_grad_norm_ = 0.;
  u_grad_norm_ = 0.;

  for (std::size_t t = 0; t < T + 1; ++t) {
    lag_mul_[t] = Vx_[t];
    lag_mul_[t].noalias() += Vxx_[t] * dxtilde_[t];
  }
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    tmp_vec_x_ = d->Lx;
    tmp_vec_x_.noalias() += d->Fx.transpose() * lag_mul_[t + 1];
    tmp_vec_x_ -= lag_mul_[t];
    if (t > 0) {
      tmp_vec_x_.noalias() += d->Gx.transpose() * y_[t];
    }
    KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
    tmp_vec_u_[t] = d->Lu;
    tmp_vec_u_[t].noalias() += d->Fu.transpose() * lag_mul_[t + 1];
    tmp_vec_u_[t].noalias() += d->Gu.transpose() * y_[t];
    KKT_ = std::max(KKT_, tmp_vec_u_[t].lpNorm<Eigen::Infinity>());
    fs_flat_.segment(t * ndx, ndx) = fs_[t];
    x_grad_norm_ += dxtilde_[t].lpNorm<1>();
    u_grad_norm_ += dutilde_[t].lpNorm<1>();
  }

  fs_flat_.tail(ndx) = fs_.back();
  const std::shared_ptr<ActionDataAbstract>& d_ter =
      problem_->get_terminalData();
  tmp_vec_x_ = d_ter->Lx;
  tmp_vec_x_ -= lag_mul_.back();
  tmp_vec_x_.noalias() += d_ter->Gx.transpose() * y_.back();
  KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, constraint_norm_);
  x_grad_norm_ += dxtilde_.back().lpNorm<1>();
  x_grad_norm_ = x_grad_norm_ / static_cast<double>(T + 1);
  u_grad_norm_ = u_grad_norm_ / static_cast<double>(T);
}

void SolverCSQP::forwardPass(const double /*stepLength*/) {
  auto profiler = crocoddyl::getProfiler().watcher("SolverCSQP::forwardPass");
  profiler.start();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    dutilde_[t] = -k_[t];
    dutilde_[t].noalias() -= K_[t] * dxtilde_[t];
    dxtilde_[t + 1] = fs_[t + 1];
    dxtilde_[t + 1].noalias() += d->Fx * dxtilde_[t];
    dxtilde_[t + 1].noalias() += d->Fu * dutilde_[t];
  }
  profiler.stop();
}

void SolverCSQP::forwardPass_without_constraints() {
  auto profiler = crocoddyl::getProfiler().watcher(
      "SolverCSQP::forwardPass_without_constraints");
  profiler.start();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    du_[t] = -k_[t];
    du_[t].noalias() -= K_[t] * dx_[t];
    dx_[t + 1] = fs_[t + 1];
    dx_[t + 1].noalias() += d->Fx * dx_[t];
    dx_[t + 1].noalias() += d->Fu * du_[t];
  }

  profiler.stop();
}

void SolverCSQP::backwardPass() {
  static auto profiler_all =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass");
  static auto profiler_Qu =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass::Qu");
  static auto profiler_Quu =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass::Quu");
  static auto profiler_Qx =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass::Qx");
  static auto profiler_Qxu =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass::Qxu");
  static auto profiler_Qxx =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass::Qxx");
  static auto profiler_Vx =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass::Vx");
  static auto profiler_Vxx =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass::Vxx");
  profiler_all.start();

  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx;
  Vxx_.back().diagonal().array() += sigma_;
  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= sigma_ * dx_.back();

  if (problem_->get_terminalModel()->get_ng()) {
    tmp_rhoGx_mat_.back().noalias() = rho_vec_.back().asDiagonal() * d_T->Gx;
    Vxx_.back().noalias() += d_T->Gx.transpose() * tmp_rhoGx_mat_.back();
    tmp_dual_cwise_.back() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }
  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];

    Vxx_fs_[t].noalias() = Vxx_[t + 1] * fs_[t + 1];
    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];

    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    profiler_Qx.start();
    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= sigma_ * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }

    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    profiler_Qx.stop();

    profiler_Qxx.start();
    Qxx_[t] = d->Lxx;
    Qxx_[t].diagonal().array() += sigma_;
    if (t > 0 && nc != 0) {
      tmp_rhoGx_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gx;
      Qxx_[t].noalias() += d->Gx.transpose() * tmp_rhoGx_mat_[t];
    }
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    profiler_Qxx.stop();

    if (nu != 0) {
      profiler_Quu.start();
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      Qu_[t] = d->Lu - sigma_ * du_[t];
      if (nc != 0) {
        Qu_[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      profiler_Qu.stop();

      profiler_Quu.start();
      Quu_[t] = d->Luu;
      Quu_[t].diagonal().array() += sigma_;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      if (nc != 0) {
        tmp_rhoGu_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gu;
        Quu_[t].noalias() += d->Gu.transpose() * tmp_rhoGu_mat_[t];
      }
      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
      }
      profiler_Quu.stop();

      profiler_Qxu.start();
      Qxu_[t] = d->Lxu;
      if (t > 0 && nc != 0) {
        Qxu_[t].noalias() += d->Gx.transpose() * tmp_rhoGu_mat_[t];
      }
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
      profiler_Qxu.stop();
    }
    computeGains(t);
    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      // Quuk_[t].noalias() = Quu_[t] * k_[t];
      profiler_Vx.start();
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      profiler_Vx.stop();
      profiler_Vxx.start();
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
      profiler_Vxx.stop();
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;
    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
    }
  }
  profiler_all.stop();
}

void SolverCSQP::backwardPass_without_constraints() {
  static auto profiler_all = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_constraints");
  static auto profiler_Qu = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_constraints::Qu");
  static auto profiler_Quu = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_constraints::Quu");
  static auto profiler_Qx = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_constraints::Qx");
  static auto profiler_Qxu = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_constraints::Qxu");
  static auto profiler_Qxx = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_constraints::Qxx");
  static auto profiler_Vx = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_constraints::Vx");
  static auto profiler_Vxx = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_constraints::Vxx");

  profiler_all.start();

  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx;
  Vx_.back() = d_T->Lx;

  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    tmp_Vx_.noalias() = Vxx_[t + 1] * fs_[t + 1];
    tmp_Vx_ += Vx_[t + 1];

    const std::size_t nu = m->get_nu();
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    profiler_Qx.start();
    Qx_[t] = d->Lx;
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    profiler_Qx.stop();
    profiler_Qxx.start();
    Qxx_[t] = d->Lxx;

    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    profiler_Qxx.stop();
    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      profiler_Qu.start();
      Qu_[t] = d->Lu;
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;

      profiler_Qu.stop();
      profiler_Quu.start();
      Quu_[t] = d->Luu;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      profiler_Quu.stop();
      profiler_Qxu.start();
      Qxu_[t] = d->Lxu;
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
      profiler_Qxu.stop();

      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
      }
    }

    computeGains(t);

    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      // Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      profiler_Vxx.start();
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
      profiler_Vxx.stop();
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;

    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
    }
  }
  profiler_all.stop();
}

void SolverCSQP::backwardPass_mt() {
  static auto profiler_all =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass_mt");
  static auto profiler_lock =
      crocoddyl::getProfiler().watcher("SolverCSQP::backwardPass_mt::lock");
  profiler_all.start();

  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx;
  Vxx_.back().diagonal().array() += sigma_;
  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= sigma_ * dx_.back();

  if (problem_->get_terminalModel()->get_ng()) {
    tmp_rhoGx_mat_.back().noalias() = rho_vec_.back().asDiagonal() * d_T->Gx;
    Vxx_.back().noalias() += d_T->Gx.transpose() * tmp_rhoGx_mat_.back();
    tmp_dual_cwise_.back() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }
  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

#pragma omp parallel for num_threads(problem_->get_nthreads())
  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();

    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= sigma_ * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }

    Qxx_[t] = d->Lxx;
    Qxx_[t].diagonal().array() += sigma_;
    if (t > 0 && nc != 0) {
      tmp_rhoGx_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gx;
      Qxx_[t].noalias() += d->Gx.transpose() * tmp_rhoGx_mat_[t];
    }

    if (nu != 0) {
      Qu_[t] = d->Lu - sigma_ * du_[t];
      if (nc != 0) {
        Qu_[t] += d->Gu.transpose() * tmp_dual_cwise_[t];
      }

      Quu_[t] = d->Luu;
      Quu_[t].diagonal().array() += sigma_;
      if (nc != 0) {
        tmp_rhoGu_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gu;
        Quu_[t].noalias() += d->Gu.transpose() * tmp_rhoGu_mat_[t];
      }
      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
      }

      Qxu_[t] = d->Lxu;
      if (t > 0 && nc != 0) {
        Qxu_[t].noalias() += d->Gx.transpose() * tmp_rhoGu_mat_[t];
      }
    }
  }

  profiler_lock.start();
  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();

    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;

    Vxx_fs_[t].noalias() = Vxx_[t + 1] * fs_[t + 1];
    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    if (nu != 0) {
      FuTVxx_p_[0].noalias() = d->Fu.transpose() * Vxx_p;
      Quu_[t].noalias() += FuTVxx_p_[0] * d->Fu;
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
    }

    computeGains(t);
    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      // Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];

      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    }
    // The commented version is theoretically slower.
    // Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_tmp_.triangularView<Eigen::Upper>() =
        (0.5 * (Vxx_[t] + Vxx_[t].transpose())).triangularView<Eigen::Upper>();
    // Vxx_[t] = Vxx_tmp_;
    Vxx_[t] = Vxx_tmp_.selfadjointView<Eigen::Upper>();

    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
    }
  }
  profiler_lock.stop();
  profiler_all.stop();
}

void SolverCSQP::backwardPass_without_rho_update() {
  static auto profiler_all = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_rho_update");
  static auto profiler_Vx = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_rho_update::Vx");
  static auto profiler_Qx = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_rho_update::Qx");
  static auto profiler_Qu = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_rho_update::Qu");
  static auto profiler_k = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_without_rho_update::k");

  profiler_all.start();

  const std::shared_ptr<crocoddyl::ActionModelAbstract>& m_T =
      problem_->get_terminalModel();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  profiler_Vx.start();
  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= sigma_ * dx_.back();

  if (m_T->get_ng()) {  // constraint model
    tmp_dual_cwise_.back() = y_.back();
    tmp_dual_cwise_.back().noalias() -= rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }
  profiler_Vx.stop();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();

    profiler_Qx.start();
    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];
    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= sigma_ * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;

    profiler_Qx.stop();

    if (nu != 0) {
      profiler_Qu.start();
      Qu_[t] = d->Lu;
      Qu_[t].noalias() -= sigma_ * du_[t];
      if (nc != 0) {
        Qu_[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      profiler_Qu.stop();
    }

    profiler_k.start();
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
    profiler_k.stop();

    profiler_Vx.start();
    Vx_[t] = Qx_[t];
    if (nu != 0) {
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    }
    profiler_Vx.stop();
  }
  profiler_all.stop();
}

void SolverCSQP::backwardPass_without_rho_update_mt() {
  static auto profiler_all = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_wo_rho_update_mt");
  static auto profiler_mt1 = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_wo_rho_update_mt::init_Qx_Qu");
  static auto profiler_pass = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_wo_rho_update_mt::update_Qx_Qu_Vx");
  static auto profiler_mt2 = crocoddyl::getProfiler().watcher(
      "SolverCSQP::backwardPass_wo_rho_update_mt::k");

  profiler_all.start();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& m_T =
      problem_->get_terminalModel();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  Vx_.back().noalias() = d_T->Lx - sigma_ * dx_.back();

  if (m_T->get_ng()) {  // constraint model
    tmp_dual_cwise_.back().noalias() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }

  profiler_mt1.start();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif  // CROCODDYL_WITH_MULTITHREADING
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();

    Qx_[t].noalias() = d->Lx - sigma_ * dx_[t];
    if (nc != 0 && (t > 0 || nu != 0)) {
      tmp_dual_cwise_[t].noalias() = y_[t] - rho_vec_[t].cwiseProduct(z_[t]);
    }
    if (nc != 0 && t > 0) {
      Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
    }

    if (nu != 0) {
      Qu_[t].noalias() = d->Lu - sigma_ * du_[t];
      if (nc != 0) {
        Qu_[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
    }
  }
  profiler_mt1.stop();

  profiler_pass.start();
  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();

    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    Vx_[t] = Qx_[t];

    if (nu != 0) {
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    }
  }
  profiler_pass.stop();

  profiler_mt2.start();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif  // CROCODDYL_WITH_MULTITHREADING
  for (std::size_t t = 0; t < T; ++t) {
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
  }
  profiler_mt2.stop();
  profiler_all.stop();
}

void SolverCSQP::update_lagrangian_parameters(const int iter) {
  static auto profiler_all = crocoddyl::getProfiler().watcher(
      "SolverCSQP::update_lagrangian_parameters");
  profiler_all.start();

  norm_primal_ = -std::numeric_limits<double>::infinity();
  norm_dual_ = -std::numeric_limits<double>::infinity();
  norm_primal_rel_ = -std::numeric_limits<double>::infinity();
  norm_dual_rel_ = -std::numeric_limits<double>::infinity();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads()) reduction( \
        max : norm_primal_, norm_dual_, norm_primal_rel_, norm_dual_rel_)
#endif  // CROCODDYL_WITH_MULTITHREADING
  for (std::size_t t = 0; t < T; ++t) {
    START_PROFILER("SolverCSQP::update_lagrangian_parameters::update");

    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    if (m->get_ng() == 0) {
      dx_[t] = dxtilde_[t];
      du_[t] = dutilde_[t];
      continue;
    }

    z_prev_[t] = z_[t];
    tmp_Cdx_Cdu_[t].noalias() = d->Gx * dxtilde_[t];
    tmp_Cdx_Cdu_[t].noalias() += d->Gu * dutilde_[t];
    z_relaxed_[t].noalias() = alpha_ * tmp_Cdx_Cdu_[t];
    z_relaxed_[t].noalias() += (1. - alpha_) * z_[t];

    tmp_dual_cwise_[t] = y_[t].cwiseProduct(inv_rho_vec_[t]);

    z_[t] = z_relaxed_[t] + tmp_dual_cwise_[t];
    z_[t] = z_[t].cwiseMax(m->get_g_lb() - d->g).cwiseMin(m->get_g_ub() - d->g);

    y_[t] += rho_vec_[t].cwiseProduct(z_relaxed_[t] - z_[t]);

    dx_[t] = dxtilde_[t];
    du_[t] = dutilde_[t];

    if (iter % rho_update_interval_ == 0) {
      if (update_rho_with_heuristic_) {
        tmp_dual_cwise_[t] = rho_vec_[t].cwiseProduct(z_[t] - z_prev_[t]);
        norm_dual_ =
            std::max(norm_dual_, tmp_dual_cwise_[t].lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(
            norm_primal_, (tmp_Cdx_Cdu_[t] - z_[t]).lpNorm<Eigen::Infinity>());

        norm_primal_rel_ = std::max(norm_primal_rel_,
                                    tmp_Cdx_Cdu_[t].lpNorm<Eigen::Infinity>());
        norm_primal_rel_ =
            std::max(norm_primal_rel_, z_[t].lpNorm<Eigen::Infinity>());
        norm_dual_rel_ =
            std::max(norm_dual_rel_, y_[t].lpNorm<Eigen::Infinity>());
      } else {
        tmp_dual_cwise_[t] = rho_vec_[t].cwiseProduct(z_[t] - z_prev_[t]);
        norm_dual_ = std::max(
            norm_dual_,
            (d->Gx.transpose() * tmp_dual_cwise_[t]).lpNorm<Eigen::Infinity>());
        norm_dual_ = std::max(
            norm_dual_,
            (d->Gu.transpose() * tmp_dual_cwise_[t]).lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(
            norm_primal_, (tmp_Cdx_Cdu_[t] - z_[t]).lpNorm<Eigen::Infinity>());

        norm_primal_rel_ = std::max(norm_primal_rel_,
                                    tmp_Cdx_Cdu_[t].lpNorm<Eigen::Infinity>());
        norm_primal_rel_ =
            std::max(norm_primal_rel_, z_[t].lpNorm<Eigen::Infinity>());
        norm_dual_rel_ =
            std::max(norm_dual_rel_,
                     (d->Gx.transpose() * y_[t]).lpNorm<Eigen::Infinity>());
        norm_dual_rel_ =
            std::max(norm_dual_rel_,
                     (d->Gu.transpose() * y_[t]).lpNorm<Eigen::Infinity>());
      }
    }
  }

  dx_.back() = dxtilde_.back();
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& m_T =
      problem_->get_terminalModel();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();
  const std::size_t nc = m_T->get_ng();

  if (nc != 0) {
    z_prev_.back() = z_.back();
    tmp_Cdx_Cdu_.back().noalias() = d_T->Gx * dxtilde_.back();
    z_relaxed_.back().noalias() = alpha_ * tmp_Cdx_Cdu_.back();
    z_relaxed_.back().noalias() += (1. - alpha_) * z_.back();

    tmp_dual_cwise_.back() = y_.back().cwiseProduct(inv_rho_vec_.back());
    z_.back() = (z_relaxed_.back() + tmp_dual_cwise_.back());
    z_.back() = z_.back()
                    .cwiseMax(m_T->get_g_lb() - d_T->g)
                    .cwiseMin(m_T->get_g_ub() - d_T->g);
    y_.back() += rho_vec_.back().cwiseProduct(z_relaxed_.back() - z_.back());

    if (iter % rho_update_interval_ == 0) {
      if (update_rho_with_heuristic_) {
        tmp_dual_cwise_.back() =
            rho_vec_.back().cwiseProduct(z_.back() - z_prev_.back());
        norm_dual_ = std::max(norm_dual_,
                              tmp_dual_cwise_.back().lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(
            norm_primal_,
            (tmp_Cdx_Cdu_.back() - z_.back()).lpNorm<Eigen::Infinity>());

        norm_primal_rel_ = std::max(
            norm_primal_rel_, tmp_Cdx_Cdu_.back().lpNorm<Eigen::Infinity>());
        norm_primal_rel_ =
            std::max(norm_primal_rel_, z_.back().lpNorm<Eigen::Infinity>());
        norm_dual_rel_ =
            std::max(norm_dual_rel_, y_.back().lpNorm<Eigen::Infinity>());
      } else {
        tmp_dual_cwise_.back() =
            rho_vec_.back().cwiseProduct(z_.back() - z_prev_.back());
        norm_dual_ =
            std::max(norm_dual_, (d_T->Gx.transpose() * tmp_dual_cwise_.back())
                                     .lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(
            norm_primal_,
            (tmp_Cdx_Cdu_.back() - z_.back()).lpNorm<Eigen::Infinity>());

        norm_primal_rel_ = std::max(
            norm_primal_rel_, tmp_Cdx_Cdu_.back().lpNorm<Eigen::Infinity>());
        norm_primal_rel_ =
            std::max(norm_primal_rel_, z_.back().lpNorm<Eigen::Infinity>());
        norm_dual_rel_ = std::max(
            norm_dual_rel_,
            (d_T->Gx.transpose() * y_.back()).lpNorm<Eigen::Infinity>());
      }
    }
  }
  profiler_all.stop();
}

double SolverCSQP::tryStep(const double steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }

  static auto profiler_tryStep =
      crocoddyl::getProfiler().watcher("SolverCSQP::tryStep");
  profiler_tryStep.start();
  cost_try_ = 0.;
  merit_try_ = 0.;
  gap_norm_try_ = 0.;
  constraint_norm_try_ = 0.;

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    m->get_state()->integrate(xs_[t], steplength * dx_[t], xs_try_[t]);
    const std::size_t nu = m->get_nu();

    if (nu != 0) {
      us_try_[t] = us_[t] + steplength * du_[t];
    }
  }

  const std::shared_ptr<crocoddyl::ActionModelAbstract>& m_ter =
      problem_->get_terminalModel();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_ter =
      problem_->get_terminalData();

  m_ter->get_state()->integrate(xs_.back(), steplength * dx_.back(),
                                xs_try_.back());

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads()) \
    reduction(+ : cost_try_, gap_norm_try_, constraint_norm_try_)
#endif  // CROCODDYL_WITH_MULTITHREADING
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    m->calc(d, xs_try_[t], us_try_[t]);
    cost_try_ += d->cost;
    m->get_state()->diff(xs_try_[t + 1], d->xnext, fs_try_[t + 1]);
    gap_norm_try_ += fs_try_[t + 1].lpNorm<1>();

    const std::size_t nc = m->get_ng();
    constraint_norm_try_ +=
        (m->get_g_lb() - d->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
    constraint_norm_try_ +=
        (d->g - m->get_g_ub()).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

    if (raiseIfNaN(cost_try_)) {
      profiler_tryStep.stop();
      throw_pretty("step_error");
    }
  }

  // Terminal state update
  m_ter->calc(d_ter, xs_try_.back());
  cost_try_ += d_ter->cost;

  const std::size_t nc = m_ter->get_ng();

  constraint_norm_try_ += (m_ter->get_g_lb() - d_ter->g)
                              .cwiseMax(Eigen::VectorXd::Zero(nc))
                              .lpNorm<1>();
  constraint_norm_try_ += (d_ter->g - m_ter->get_g_ub())
                              .cwiseMax(Eigen::VectorXd::Zero(nc))
                              .lpNorm<1>();

  if (mu_dynamic_ < 0. || mu_constraint_ < 0.) {
    merit_try_ = cost_try_ + lag_mul_inf_norm_coef_ * lag_mul_inf_norm_ *
                                 (gap_norm_try_ + constraint_norm_try_);

  } else {
    merit_try_ = cost_try_ + mu_dynamic_ * gap_norm_try_ +
                 mu_constraint_ * constraint_norm_try_;
  }

  if (raiseIfNaN(cost_try_)) {
    profiler_tryStep.stop();
    throw_pretty("step_error");
  }

  profiler_tryStep.stop();

  return merit_try_;
}

void SolverCSQP::printQPCallbacks(const int iter) {
  std::cout << "Iters " << iter;
  std::cout << " norm_primal = " << std::scientific << std::setprecision(4)
            << norm_primal_;
  std::cout << " norm_primal_tol = " << std::scientific << std::setprecision(4)
            << norm_primal_tolerance_;
  std::cout << " norm_dual =  " << std::scientific << std::setprecision(4)
            << norm_dual_;
  std::cout << " norm_dual_tol = " << std::scientific << std::setprecision(4)
            << norm_dual_tolerance_;
  std::cout << std::endl;
  std::cout << std::flush;
}

void SolverCSQP::setQPCallbacks(const bool inQPCallbacks) {
  with_qp_callbacks_ = inQPCallbacks;
}

}  // namespace mim_solvers
