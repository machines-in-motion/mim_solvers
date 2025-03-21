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

#include <crocoddyl/core/utils/exception.hpp>
#include <iomanip>
#include <iostream>

#include "mim_solvers/sqp.hpp"

using namespace crocoddyl;

namespace mim_solvers {

SolverSQP::SolverSQP(std::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverDDP(problem) {
  const std::size_t T = this->problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  // std::cout << "ndx" << ndx << std::endl;
  fs_try_.resize(T + 1);
  fs_flat_.resize(ndx * (T + 1));
  fs_flat_.setZero();
  dx_.resize(T + 1);
  lag_mul_.resize(T + 1);
  du_.resize(T);
  gap_list_.resize(filter_size_);
  cost_list_.resize(filter_size_);
  tmp_vec_x_.resize(ndx);
  tmp_vec_u_.resize(T);
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    dx_[t].resize(ndx);
    du_[t].resize(nu);
    fs_try_[t].resize(ndx);
    lag_mul_[t].resize(ndx);
    lag_mul_[t].setZero();
    dx_[t].setZero();
    du_[t] = Eigen::VectorXd::Zero(nu);
    fs_try_[t] = Eigen::VectorXd::Zero(ndx);
    tmp_vec_u_[t].resize(nu);
  }
  lag_mul_.back().resize(ndx);
  lag_mul_.back().setZero();
  dx_.back().resize(ndx);
  dx_.back().setZero();
  fs_try_.back().resize(ndx);
  fs_try_.back() = Eigen::VectorXd::Zero(ndx);

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

SolverSQP::~SolverSQP() {}

bool SolverSQP::solve(const std::vector<Eigen::VectorXd>& init_xs,
                      const std::vector<Eigen::VectorXd>& init_us,
                      const std::size_t maxiter, const bool is_feasible,
                      const double reginit) {
  START_PROFILER("SolverSQP::solve");
  (void)is_feasible;

  if (problem_->is_updated()) {
    resizeData();
  }
  setCandidate(init_xs, init_us, false);
  xs_[0] = problem_->get_x0();  // Otherwise xs[0] is overwritten by init_xs
                                // inside setCandidate()
  xs_try_[0] =
      problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible

  if (std::isnan(reginit)) {
    preg_ = reg_min_;
    dreg_ = reg_min_;
  } else {
    preg_ = reginit;
    dreg_ = reginit;
  }

  bool recalcDiff = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    recalcDiff = true;

    while (true) {
      try {
        computeDirection(recalcDiff);
      } catch (std::exception& e) {
        recalcDiff = false;
        increaseRegularization();
        if (preg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }

    // KKT termination criteria
    checkKKTConditions();

    // Perform callbacks
    const std::size_t n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      mim_solvers::CallbackAbstract& callback = *callbacks_[c];
      callback(*this, "SQP");
    }

    if (KKT_ <= termination_tol_) {
      STOP_PROFILER("SolverSQP::solve");
      return true;
    }

    gap_list_.push_back(gap_norm_);
    cost_list_.push_back(cost_);

    // We need to recalculate the derivatives when the step length passes
    for (std::vector<double>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
      steplength_ = *it;
      try {
        merit_try_ = tryStep(steplength_);
      } catch (std::exception& e) {
        continue;
      }
      // Filter line search criteria
      // Equivalent to heuristic cost_ > cost_try_ || gap_norm_ > gap_norm_try_
      // when filter_size=1
      if (use_filter_line_search_) {
        is_worse_than_memory_ = false;
        std::size_t count = 0.;
        while (count < filter_size_ && is_worse_than_memory_ == false and
               count <= iter_) {
          is_worse_than_memory_ =
              cost_list_[filter_size_ - 1 - count] <= cost_try_ &&
              gap_list_[filter_size_ - 1 - count] <= gap_norm_try_;
          count++;
        }
        if (is_worse_than_memory_ == false) {
          setCandidate(xs_try_, us_try_, false);
          recalcDiff = true;
          break;
        }
      }
      // Line-search criteria using merit function
      else {
        if (merit_ > merit_try_) {
          setCandidate(xs_try_, us_try_, false);
          recalcDiff = true;
          break;
        }
      }
    }

    if (steplength_ > th_stepdec_) {
      decreaseRegularization();
    }
    if (steplength_ <= th_stepinc_) {
      increaseRegularization();
      if (preg_ == reg_max_) {
        STOP_PROFILER("SolverSQP::solve");
        return false;
      }
    }
  }

  if (extra_iteration_for_last_kkt_) {
    recalcDiff = true;

    while (true) {
      try {
        computeDirection(recalcDiff);
      } catch (std::exception& e) {
        recalcDiff = false;
        increaseRegularization();
        if (preg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }

    // KKT termination criteria
    checkKKTConditions();

    // Perform callbacks
    const std::size_t n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      mim_solvers::CallbackAbstract& callback = *callbacks_[c];
      callback(*this, "SQP");
    }

    if (KKT_ <= termination_tol_) {
      STOP_PROFILER("SolverSQP::solve");
      return true;
    }
  }

  STOP_PROFILER("SolverSQP::solve");
  return false;
}

void SolverSQP::computeDirection(const bool recalcDiff) {
  START_PROFILER("SolverSQP::computeDirection");
  if (recalcDiff) {
    cost_ = calcDiff();
  }
  gap_norm_ = 0;
  const std::size_t T = problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    gap_norm_ += fs_[t].lpNorm<1>();
  }
  gap_norm_ += fs_.back().lpNorm<1>();

  merit_ = cost_ + mu_dynamic_ * gap_norm_;

  backwardPass();
  forwardPass();

  STOP_PROFILER("SolverSQP::computeDirection");
}

void SolverSQP::checkKKTConditions() {
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    tmp_vec_x_ = d->Lx;
    tmp_vec_x_.noalias() += d->Fx.transpose() * lag_mul_[t + 1];
    tmp_vec_x_ -= lag_mul_[t];
    tmp_vec_u_[t] = d->Lu;
    tmp_vec_u_[t].noalias() += d->Fu.transpose() * lag_mul_[t + 1];
    KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
    KKT_ = std::max(KKT_, tmp_vec_u_[t].lpNorm<Eigen::Infinity>());
    fs_flat_.segment(t * ndx, ndx) = fs_[t];
  }
  fs_flat_.tail(ndx) = fs_.back();
  const std::shared_ptr<ActionDataAbstract>& d_ter =
      problem_->get_terminalData();
  tmp_vec_x_ = d_ter->Lx;
  tmp_vec_x_ -= lag_mul_.back();
  KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
}

void SolverSQP::forwardPass(const double stepLength) {
  (void)stepLength;

  START_PROFILER("SolverSQP::forwardPass");
  x_grad_norm_ = 0;
  u_grad_norm_ = 0;

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    tmp_vec_x_ = dx_[t] - fs_[t];
    lag_mul_[t].noalias() = Vxx_[t] * tmp_vec_x_;
    lag_mul_[t].noalias() += Vx_[t];
    du_[t].noalias() = -K_[t] * dx_[t];
    du_[t].noalias() -= k_[t];
    dx_[t + 1].noalias() = fs_[t + 1];
    dx_[t + 1].noalias() += d->Fu * du_[t];
    dx_[t + 1].noalias() += d->Fx * dx_[t];
    x_grad_norm_ +=
        dx_[t]
            .lpNorm<1>();  // assuming that there is no gap in the initial state
    u_grad_norm_ += du_[t].lpNorm<1>();
  }

  lag_mul_.back() = Vx_.back();
  tmp_vec_x_ = dx_.back() - fs_.back();
  lag_mul_.back().noalias() += Vxx_.back() * tmp_vec_x_;
  x_grad_norm_ +=
      dx_.back()
          .lpNorm<1>();  // assuming that there is no gap in the initial state
  x_grad_norm_ = x_grad_norm_ / (double)(T + 1);
  u_grad_norm_ = u_grad_norm_ / (double)T;
  STOP_PROFILER("SolverSQP::forwardPass");
}

double SolverSQP::tryStep(const double steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  START_PROFILER("SolverSQP::tryStep");
  cost_try_ = 0.;
  merit_try_ = 0;
  gap_norm_try_ = 0;

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();

    m->get_state()->integrate(xs_[t], steplength * dx_[t], xs_try_[t]);
    if (nu != 0) {
      us_try_[t].noalias() = us_[t];
      us_try_[t].noalias() += steplength * du_[t];
    }
    m->calc(d, xs_try_[t], us_try_[t]);
    cost_try_ += d->cost;

    if (t > 0) {
      const std::shared_ptr<ActionDataAbstract>& d_prev = datas[t - 1];
      m->get_state()->diff(xs_try_[t], d_prev->xnext, fs_try_[t - 1]);
      gap_norm_try_ += fs_try_[t - 1].lpNorm<1>();
    }

    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverSQP::tryStep");
      throw_pretty("step_error");
    }
  }

  // Terminal state update
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& m_ter =
      problem_->get_terminalModel();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_ter =
      problem_->get_terminalData();
  m_ter->get_state()->integrate(xs_.back(), steplength * dx_.back(),
                                xs_try_.back());
  m_ter->calc(d_ter, xs_try_.back());
  cost_try_ += d_ter->cost;

  const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[T - 1];
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[T - 1];

  m->get_state()->diff(xs_try_.back(), d->xnext, fs_try_[T - 1]);
  gap_norm_try_ += fs_try_[T - 1].lpNorm<1>();

  merit_try_ = cost_try_ + mu_dynamic_ * gap_norm_try_;

  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverSQP::tryStep");
    throw_pretty("step_error");
  }

  STOP_PROFILER("SolverSQP::tryStep");

  return merit_try_;
}

// double SolverSQP::get_th_acceptnegstep() const { return th_acceptnegstep_; }

// void SolverSQP::set_th_acceptnegstep(const double th_acceptnegstep) {
//   if (0. > th_acceptnegstep) {
//     throw_pretty("Invalid argument: "
//                  << "th_acceptnegstep value has to be positive.");
//   }
//   th_acceptnegstep_ = th_acceptnegstep;
// }

}  // namespace mim_solvers
