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

  // Initialize filter lists
  constraint_list_.resize(filter_size_);
  gap_list_.resize(filter_size_);
  cost_list_.resize(filter_size_);

  fs_flat_.resize(ndx * (T + 1));
  fs_flat_.setZero();

  // Allocate trial trajectories
  xs_try_.resize(T + 1);
  us_try_.resize(T);
  lag_mul_.resize(T + 1);
  fs_try_.resize(T + 1);

  tmp_vec_x_.resize(ndx);
  tmp_vec_x_.setZero();
  tmp_vec_u_.resize(T);

  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();

  std::size_t n_eq_crocoddyl = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    n_eq_crocoddyl += model->get_nh();

    xs_try_[t] = model->get_state()->zero();
    us_try_[t] = Eigen::VectorXd::Zero(nu);
    lag_mul_[t].resize(ndx);
    lag_mul_[t].setZero();
    fs_try_[t].resize(ndx);
    fs_try_[t].setZero();
    tmp_vec_u_[t].resize(nu);
    tmp_vec_u_[t].setZero();
  }

  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();
  lag_mul_.back().resize(ndx);
  lag_mul_.back().setZero();
  fs_try_.back().resize(ndx);
  fs_try_.back().setZero();

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

  // Create the inner QP solver
  qp_solver_ = std::make_unique<SolverOSQP_QP>(problem, this);
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
  // Otherwise xs[0] is overwritten by init_xs inside setCandidate()
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
    qp_solver_->reset_rho_vec();
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
    if (iter_ == 0 && !qp_solver_->get_reset_rho()) {
      qp_solver_->reset_rho_vec();
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
    if (qp_solver_->get_qp_iters() == 0) {
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
      for (const auto& lm : lag_mul_) {
        lag_mul_inf_norm_ =
            std::max(lag_mul_inf_norm_, lm.lpNorm<Eigen::Infinity>());
      }
      for (const auto& y : qp_solver_->get_y()) {
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
  START_PROFILER("SolverCSQP::computeDirection");

  // Coordinate timeout with QP solver
  qp_solver_->set_start_time(start_time_);
  qp_solver_->set_max_solve_time(max_solve_time_);

  // Delegate to inner QP solver
  qp_solver_->computeDirection();

  // Check if QP solver timed out
  if (qp_solver_->get_max_solve_time_reached()) {
    max_solve_time_reached_ = true;
  }

  STOP_PROFILER("SolverCSQP::computeDirection");
}

void SolverCSQP::checkKKTConditions() {
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();
  x_grad_norm_ = 0.;
  u_grad_norm_ = 0.;

  const std::vector<Eigen::VectorXd>& dxtilde = qp_solver_->get_dx_tilde();
  const std::vector<Eigen::VectorXd>& dutilde = qp_solver_->get_du_tilde();
  const std::vector<Eigen::VectorXd>& y = qp_solver_->get_y();

  for (std::size_t t = 0; t < T + 1; ++t) {
    lag_mul_[t] = Vx_[t];
    lag_mul_[t].noalias() += Vxx_[t] * dxtilde[t];
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
      tmp_vec_x_.noalias() += d->Gx.transpose() * y[t];
    }
    KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
    tmp_vec_u_[t] = d->Lu;
    tmp_vec_u_[t].noalias() += d->Fu.transpose() * lag_mul_[t + 1];
    tmp_vec_u_[t].noalias() += d->Gu.transpose() * y[t];
    KKT_ = std::max(KKT_, tmp_vec_u_[t].lpNorm<Eigen::Infinity>());
    fs_flat_.segment(t * ndx, ndx) = fs_[t];
    x_grad_norm_ += dxtilde[t].lpNorm<1>();
    u_grad_norm_ += dutilde[t].lpNorm<1>();
  }

  fs_flat_.tail(ndx) = fs_.back();
  const std::shared_ptr<ActionDataAbstract>& d_ter =
      problem_->get_terminalData();
  tmp_vec_x_ = d_ter->Lx;
  tmp_vec_x_ -= lag_mul_.back();
  tmp_vec_x_.noalias() += d_ter->Gx.transpose() * y.back();
  KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, constraint_norm_);
  x_grad_norm_ += dxtilde.back().lpNorm<1>();
  x_grad_norm_ = x_grad_norm_ / static_cast<double>(T + 1);
  u_grad_norm_ = u_grad_norm_ / static_cast<double>(T);
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

  const std::vector<Eigen::VectorXd>& dx = qp_solver_->get_dx();
  const std::vector<Eigen::VectorXd>& du = qp_solver_->get_du();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    m->get_state()->integrate(xs_[t], steplength * dx[t], xs_try_[t]);
    const std::size_t nu = m->get_nu();

    if (nu != 0) {
      us_try_[t] = us_[t] + steplength * du[t];
    }
  }

  const std::shared_ptr<crocoddyl::ActionModelAbstract>& m_ter =
      problem_->get_terminalModel();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_ter =
      problem_->get_terminalData();

  m_ter->get_state()->integrate(xs_.back(), steplength * dx.back(),
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

}  // namespace mim_solvers
