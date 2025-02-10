///////////////////////////////////////////////////////////////////////////////
//
// This file is a modified version of SolverDDP from the Crocoddyl library
// This modified version is used for benchmarking purposes only
// Original file :
// https://github.com/loco-3d/crocoddyl/blob/devel/src/core/solvers/fddp.cpp
//
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

#include "mim_solvers/fddp.hpp"

using namespace crocoddyl;

namespace mim_solvers {

SolverFDDP::SolverFDDP(std::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverDDP(problem), dg_(0), dq_(0), dv_(0), th_acceptnegstep_(2) {
  std::cerr << "Warning: Do not use mim_solvers.SolverFDDP !!! " << std::endl;
  std::cerr << "It may differ significantly from its Crocoddyl counterpart. "
            << std::endl;
  std::cerr << "This class served only as a development tool and will be "
               "REMOVED in future releases."
            << std::endl;
  const std::size_t T = this->problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  fs_try_.resize(T + 1);
  lag_mul_.resize(T + 1);
  fs_flat_.resize(ndx * (T + 1));
  fs_flat_.setZero();
  gap_list_.resize(filter_size_);
  cost_list_.resize(filter_size_);
  for (std::size_t t = 0; t < T; ++t) {
    lag_mul_[t].resize(ndx);
    lag_mul_[t].setZero();
    fs_try_[t].resize(ndx);
    fs_try_[t] = Eigen::VectorXd::Zero(ndx);
  }
  lag_mul_.back().resize(ndx);
  lag_mul_.back().setZero();
  fs_try_.back().resize(ndx);
  fs_try_.back() = Eigen::VectorXd::Zero(ndx);
}

SolverFDDP::~SolverFDDP() {}

bool SolverFDDP::solve(const std::vector<Eigen::VectorXd>& init_xs,
                       const std::vector<Eigen::VectorXd>& init_us,
                       const std::size_t maxiter, const bool is_feasible,
                       const double reginit) {
  START_PROFILER("SolverFDDP::solve");
  if (problem_->is_updated()) {
    resizeData();
  }
  xs_try_[0] =
      problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible
  setCandidate(init_xs, init_us, is_feasible);

  if (std::isnan(reginit)) {
    preg_ = reg_min_;
    dreg_ = reg_min_;
  } else {
    preg_ = reginit;
    dreg_ = reginit;
  }
  was_feasible_ = false;
  bool recalcDiff = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
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
    updateExpectedImprovement();

    // KKT termination criteria
    checkKKTConditions();
    if (KKT_ <= termination_tol_) {
      STOP_PROFILER("SolverFDDP::solve");
      return true;
    }

    gap_list_.push_back(gap_norm_);
    cost_list_.push_back(cost_);

    // We need to recalculate the derivatives when the step length passes
    recalcDiff = false;
    for (std::vector<double>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
      steplength_ = *it;

      try {
        dV_ = tryStep(steplength_);
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
      // Using expectation model
      else {
        expectedImprovement();
        dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

        if (dVexp_ >= 0) {  // descend direction
          if (d_[0] < th_grad_ || dV_ > th_acceptstep_ * dVexp_) {
            was_feasible_ = is_feasible_;
            setCandidate(xs_try_, us_try_,
                         (was_feasible_) || (steplength_ == 1));
            cost_ = cost_try_;
            recalcDiff = true;
            break;
          }
        } else {  // reducing the gaps by allowing a small increment in the cost
                  // value
          if (dV_ > th_acceptnegstep_ * dVexp_) {
            was_feasible_ = is_feasible_;
            setCandidate(xs_try_, us_try_,
                         (was_feasible_) || (steplength_ == 1));
            cost_ = cost_try_;
            recalcDiff = true;
            break;
          }
        }
      }
    }

    if (steplength_ > th_stepdec_) {
      decreaseRegularization();
    }
    if (steplength_ <= th_stepinc_) {
      increaseRegularization();
      if (preg_ == reg_max_) {
        STOP_PROFILER("SolverFDDP::solve");
        return false;
      }
    }
    stoppingCriteria();

    // Perform callbacks
    const std::size_t n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      mim_solvers::CallbackAbstract& callback = *callbacks_[c];
      callback(*this, "SQP");
    }

    // std::cout << "KKT = " << KKT_ << std::endl;
  }
  STOP_PROFILER("SolverFDDP::solve");
  return false;
}

void SolverFDDP::computeDirection(const bool recalcDiff) {
  START_PROFILER("SolverFDDP::computeDirection");
  if (recalcDiff) {
    calcDiff();
  }

  gap_norm_ = 0;
  const std::size_t T = problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    gap_norm_ += fs_[t].lpNorm<1>();
  }
  gap_norm_ += fs_.back().lpNorm<1>();

  backwardPass();
  STOP_PROFILER("SolverFDDP::computeDirection");
}

void SolverFDDP::checkKKTConditions() {
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    KKT_ = std::max(KKT_,
                    (d->Lx + d->Fx.transpose() * lag_mul_[t + 1] - lag_mul_[t])
                        .lpNorm<Eigen::Infinity>());
    KKT_ = std::max(KKT_, (d->Lu + d->Fu.transpose() * lag_mul_[t + 1])
                              .lpNorm<Eigen::Infinity>());
    fs_flat_.segment(t * ndx, ndx) = fs_[t];
  }
  fs_flat_.tail(ndx) = fs_.back();
  const std::shared_ptr<ActionDataAbstract>& d_ter =
      problem_->get_terminalData();
  KKT_ =
      std::max(KKT_, (d_ter->Lx - lag_mul_.back()).lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
}

const Eigen::Vector2d& SolverFDDP::expectedImprovement() {
  dv_ = 0;
  const std::size_t T = this->problem_->get_T();
  if (!is_feasible_) {
    // NB: The dimension of vectors xs_try_ and xs_ are T+1, whereas the
    // dimension of dx_ is T. Here, we are re-using the final element of dx_ for
    // the computation of the difference at the terminal node. Using the access
    // iterator back() this re-use of the final element is fine. Cf. the
    // discussion at https://github.com/loco-3d/crocoddyl/issues/1022
    problem_->get_terminalModel()->get_state()->diff(xs_try_.back(), xs_.back(),
                                                     dx_.back());
    fTVxx_p_.noalias() = Vxx_.back() * dx_.back();
    dv_ -= fs_.back().dot(fTVxx_p_);
    const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
        problem_->get_runningModels();

    for (std::size_t t = 0; t < T; ++t) {
      models[t]->get_state()->diff(xs_try_[t], xs_[t], dx_[t]);
      fTVxx_p_.noalias() = Vxx_[t] * dx_[t];
      dv_ -= fs_[t].dot(fTVxx_p_);
    }
  }
  d_[0] = dg_ + dv_;
  d_[1] = dq_ - 2 * dv_;
  return d_;
}

void SolverFDDP::updateExpectedImprovement() {
  dg_ = 0;
  dq_ = 0;
  const std::size_t T = this->problem_->get_T();
  if (!is_feasible_) {
    dg_ -= Vx_.back().dot(fs_.back());
    fTVxx_p_.noalias() = Vxx_.back() * fs_.back();
    dq_ += fs_.back().dot(fTVxx_p_);
  }
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t nu = models[t]->get_nu();
    if (nu != 0) {
      dg_ += Qu_[t].dot(k_[t]);
      dq_ -= k_[t].dot(Quuk_[t]);
    }
    if (!is_feasible_) {
      dg_ -= Vx_[t].dot(fs_[t]);
      fTVxx_p_.noalias() = Vxx_[t] * fs_[t];
      dq_ += fs_[t].dot(fTVxx_p_);
    }
  }
}

void SolverFDDP::forwardPass(const double steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  START_PROFILER("SolverFDDP::forwardPass");
  cost_try_ = 0.;
  gap_norm_try_ = 0;
  xnext_ = problem_->get_x0();
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  if ((is_feasible_) || (steplength == 1)) {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& m = models[t];
      const std::shared_ptr<ActionDataAbstract>& d = datas[t];
      const std::size_t nu = m->get_nu();
      // compute Lagrange multipliers for KKT condition check
      lag_mul_[t].noalias() = Vxx_[t] * dx_[t] + Vx_[t];
      xs_try_[t] = xnext_;
      m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
      if (nu != 0) {
        us_try_[t].noalias() = us_[t] - k_[t] * steplength - K_[t] * dx_[t];
        m->calc(d, xs_try_[t], us_try_[t]);
      } else {
        m->calc(d, xs_try_[t]);
      }
      xnext_ = d->xnext;
      cost_try_ += d->cost;

      m->get_state()->diff(xs_try_[t], d->xnext, fs_try_[t]);
      gap_norm_try_ += fs_try_[t].lpNorm<1>();

      if (raiseIfNaN(cost_try_)) {
        STOP_PROFILER("SolverFDDP::forwardPass");
        throw_pretty("forward_error");
      }
      if (raiseIfNaN(xnext_.lpNorm<Eigen::Infinity>())) {
        STOP_PROFILER("SolverFDDP::forwardPass");
        throw_pretty("forward_error");
      }
    }

    const std::shared_ptr<ActionModelAbstract>& m =
        problem_->get_terminalModel();
    const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
    lag_mul_.back() = Vxx_.back() * dx_.back() + Vx_.back();
    xs_try_.back() = xnext_;
    m->calc(d, xs_try_.back());
    cost_try_ += d->cost;

    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverFDDP::forwardPass");
      throw_pretty("forward_error");
    }
  } else {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& m = models[t];
      const std::shared_ptr<ActionDataAbstract>& d = datas[t];
      const std::size_t nu = m->get_nu();
      lag_mul_[t].noalias() = Vxx_[t] * (dx_[t] - fs_[t]) + Vx_[t];
      m->get_state()->integrate(xnext_, fs_[t] * (steplength - 1), xs_try_[t]);
      m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
      if (nu != 0) {
        us_try_[t].noalias() = us_[t] - k_[t] * steplength - K_[t] * dx_[t];
        m->calc(d, xs_try_[t], us_try_[t]);
      } else {
        m->calc(d, xs_try_[t]);
      }
      xnext_ = d->xnext;
      cost_try_ += d->cost;

      m->get_state()->diff(xs_try_[t], d->xnext, fs_try_[t]);
      gap_norm_try_ += fs_try_[t].lpNorm<1>();

      if (raiseIfNaN(cost_try_)) {
        STOP_PROFILER("SolverFDDP::forwardPass");
        throw_pretty("forward_error");
      }
      if (raiseIfNaN(xnext_.lpNorm<Eigen::Infinity>())) {
        STOP_PROFILER("SolverFDDP::forwardPass");
        throw_pretty("forward_error");
      }
    }

    const std::shared_ptr<ActionModelAbstract>& m =
        problem_->get_terminalModel();
    const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
    lag_mul_.back() = Vxx_.back() * (dx_.back() - fs_.back()) + Vx_.back();
    m->get_state()->integrate(xnext_, fs_.back() * (steplength - 1),
                              xs_try_.back());
    m->calc(d, xs_try_.back());
    cost_try_ += d->cost;

    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverFDDP::forwardPass");
      throw_pretty("forward_error");
    }
  }
  STOP_PROFILER("SolverFDDP::forwardPass");
}

double SolverFDDP::get_th_acceptnegstep() const { return th_acceptnegstep_; }

void SolverFDDP::set_th_acceptnegstep(const double th_acceptnegstep) {
  if (0. > th_acceptnegstep) {
    throw_pretty(
        "Invalid argument: " << "th_acceptnegstep value has to be positive.");
  }
  th_acceptnegstep_ = th_acceptnegstep;
}

}  // namespace mim_solvers
