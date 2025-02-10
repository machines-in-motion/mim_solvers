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

#include <iomanip>
#include <iostream>

#include "crocoddyl/core/utils/exception.hpp"
#include "mim_solvers/csqp_proxqp.hpp"

using namespace crocoddyl;

namespace mim_solvers {

SolverPROXQP::SolverPROXQP(std::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverDDP(problem) {
  // std::cout << tmp << std::endl;
  const std::size_t T = this->problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  constraint_list_.resize(filter_size_);
  gap_list_.resize(filter_size_);
  cost_list_.resize(filter_size_);
  fs_try_.resize(T + 1);
  fs_flat_.resize(ndx * (T + 1));
  fs_flat_.setZero();
  lag_mul_.resize(T + 1);
  y_.resize(T + 1);
  du_.resize(T);
  dx_.resize(T + 1);
  du_.resize(T);

  xs_try_.resize(T + 1);
  us_try_.resize(T);

  std::size_t n_eq_crocoddyl = 0;

  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  // We allow nx and nu to change with time
  n_vars = 0;
  n_eq = 0;
  for (std::size_t t = 0; t < T; ++t) {
    if (t < T - 1) {
      n_vars += models[t + 1]->get_state()->get_nx();
      n_eq += models[t + 1]->get_state()->get_nx();
    }
    n_vars += models[t]->get_nu();
  }
  n_vars += problem_->get_terminalModel()->get_state()->get_nx();
  n_eq += problem_->get_terminalModel()->get_state()->get_nx();
  P_.resize(n_vars, n_vars);
  P_.setZero();
  q_.resize(n_vars), q_.setZero();
  A_.resize(n_eq, n_vars);
  A_.setZero();
  b_.resize(n_eq);
  b_.setZero();

  Psp_.resize(n_vars, n_vars);
  Asp_.resize(n_eq, n_vars);

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    xs_try_[t] = model->get_state()->zero();
    us_try_[t] = Eigen::VectorXd::Zero(nu);
    fs_try_[t].resize(ndx);
    fs_try_[t] = Eigen::VectorXd::Zero(ndx);
    lag_mul_[t].resize(ndx);
    lag_mul_[t].setZero();
    dx_[t].resize(ndx);
    du_[t].resize(nu);
    dx_[t].setZero();
    du_[t] = Eigen::VectorXd::Zero(nu);

    // Constraint Models
    int nc = model->get_ng();
    n_eq_crocoddyl += model->get_nh();
    y_[t].resize(nc);
    y_[t].setZero();
    n_in += nc;
  }

  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();

  lag_mul_.back().resize(ndx);
  lag_mul_.back().setZero();
  dx_.back().resize(ndx);
  dx_.back().setZero();
  fs_try_.back().resize(ndx);
  fs_try_.back() = Eigen::VectorXd::Zero(ndx);

  // Constraint Models
  int nc = problem_->get_terminalModel()->get_ng();

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

  y_.back().resize(nc);
  y_.back().setZero();
  n_in += nc;
  // n_eq = T* nx;

  C_.resize(n_in, n_vars);
  C_.setZero();
  l_.resize(n_in);
  l_.setZero();
  u_.resize(n_in);
  u_.setZero();

  Csp_.resize(n_in, n_vars);

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

SolverPROXQP::~SolverPROXQP() {}

bool SolverPROXQP::solve(const std::vector<Eigen::VectorXd>& init_xs,
                         const std::vector<Eigen::VectorXd>& init_us,
                         const std::size_t maxiter, const bool is_feasible,
                         const double reginit) {
  (void)is_feasible;
  (void)reginit;

  START_PROFILER("SolverPROXQP::solve");
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

  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    was_feasible_ = false;
    bool recalcDiff = true;

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
    if (KKT_ <= termination_tol_) {
      if (with_callbacks_) {
        printCallbacks();
      }
      STOP_PROFILER("SolverPROXQP::solve");
      return true;
    }

    constraint_list_.push_back(constraint_norm_);
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
              gap_list_[filter_size_ - 1 - count] <= gap_norm_try_ &&
              constraint_list_[filter_size_ - 1 - count] <=
                  constraint_norm_try_;
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
        STOP_PROFILER("SolverPROXQP::solve");
        return false;
      }
    }
    if (with_callbacks_) {
      printCallbacks();
    }
  }
  STOP_PROFILER("SolverPROXQP::solve");
  return false;
}

void SolverPROXQP::calc(const bool recalc) {
  if (recalc) {
    problem_->calc(xs_, us_);
    cost_ = problem_->calcDiff(xs_, us_);
  }

  gap_norm_ = 0;
  constraint_norm_ = 0;
  // double infty = std::numeric_limits<double>::infinity();
  double nin_count = 0;

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();

  double index_u = n_eq;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    const int nx = m->get_state()->get_nx();
    const int nu = m->get_nu();

    m->get_state()->diff(xs_[t + 1], d->xnext, fs_[t + 1]);

    gap_norm_ += fs_[t + 1].lpNorm<1>();

    int nc = m->get_ng();
    auto lb = m->get_g_lb();
    auto ub = m->get_g_ub();
    constraint_norm_ +=
        (lb - d->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
    constraint_norm_ +=
        (d->g - ub).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

    // creating matrices
    if (t > 0) {
      const double index_x = (t - 1) * nx;
      P_.middleCols(index_x, nx).middleRows(index_x, nx) = d->Lxx;
      P_.middleCols(index_u, nu).middleRows(index_x, nx) = d->Lxu;
      P_.middleCols(index_x, nx).middleRows(index_u, nu) = d->Lxu.transpose();
      q_.segment(index_x, nx) = d->Lx;

      A_.middleCols((t - 1) * nx, nx).middleRows(t * nx, nx) = -d->Fx;

      if (nc != 0) {
        C_.middleCols((t - 1) * nx, nx).middleRows(nin_count, nc) = d->Gx;
      }
    }
    if (nc != 0) {
      C_.middleCols(T * nx + (t)*nu, nu).middleRows(nin_count, nc) = d->Gu;
      l_.segment(nin_count, nc) = lb - d->g;
      u_.segment(nin_count, nc) = ub - d->g;
    }
    nin_count += nc;

    P_.middleCols(index_u, nu).middleRows(index_u, nu) = d->Luu;
    q_.segment(index_u, nu) = d->Lu;

    A_.middleCols(index_u, nu).middleRows(t * nx, nx) = -d->Fu;
    // make faster
    A_.middleCols(t * nx, nx).middleRows(t * nx, nx) =
        Eigen::MatrixXd::Identity(nx, nx);
    b_.segment(t * nx, nx) = fs_[t + 1];

    index_u += nu;
  }

  const std::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  const std::shared_ptr<ActionModelAbstract>& m_T =
      problem_->get_terminalModel();
  int nc = m_T->get_ng();
  auto lb = m_T->get_g_lb();
  auto ub = m_T->get_g_ub();
  const int nx = m_T->get_state()->get_nx();

  P_.middleCols((T - 1) * nx, nx).middleRows((T - 1) * nx, nx) = d_T->Lxx;
  q_.segment((T - 1) * nx, nx) = d_T->Lx;

  if (nc != 0) {
    C_.middleCols((T - 1) * nx, nx).middleRows(nin_count, nc) = d_T->Gx;
    l_.segment(nin_count, nc) = lb - d_T->g;
    u_.segment(nin_count, nc) = ub - d_T->g;
    nin_count += nc;
  }

  constraint_norm_ +=
      (lb - d_T->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
  constraint_norm_ +=
      (d_T->g - ub).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

  merit_ = cost_ + mu_dynamic_ * gap_norm_ + mu_constraint_ * constraint_norm_;

  Asp_ = A_.sparseView();
  Csp_ = C_.sparseView();
  Psp_ = P_.sparseView();
}

void SolverPROXQP::computeDirection(const bool recalcDiff) {
  START_PROFILER("SolverPROXQP::computeDirection");
  const std::size_t T = problem_->get_T();
  if (recalcDiff) {
    calc(recalcDiff);
  }
  checkKKTConditions();

  // proxsuite::proxqp::dense::QP<double> qp_(n_vars, n_eq, n_in);
  // qp_.init(P_, q_, A_, b_, C_, l_, u_);

  proxsuite::proxqp::sparse::QP<double, long long> qp_(n_vars, n_eq, n_in);
  qp_.init(Psp_, q_, Asp_, b_, Csp_, l_, u_);

  qp_.settings.eps_abs = eps_abs_;
  qp_.settings.eps_rel = eps_rel_;
  qp_.settings.max_iter = max_qp_iters_;
  qp_.settings.max_iter_in = max_qp_iters_;
  qp_.solve();
  qp_iters_ = qp_.results.info.iter;
  norm_primal_ = qp_.results.info.pri_res;
  norm_dual_ = qp_.results.info.dua_res;

  // std::cout << "ProxQP iters " << qp_iters_ << " norm_primal=" <<
  // norm_primal_ << " norm_dual= " << norm_dual_ << std::endl;
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  x_grad_norm_ = 0;
  u_grad_norm_ = 0;
  double nin_count = 0;
  double index_u = n_eq;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const int nx = m->get_state()->get_nx();
    const int nu = m->get_nu();
    int nc = m->get_ng();

    dx_[t + 1] = qp_.results.x.segment(t * nx, nx);
    // double index_u = T *nx + t * nu;
    du_[t] = qp_.results.x.segment(index_u, nu);
    x_grad_norm_ +=
        dx_[t + 1]
            .lpNorm<1>();  // assuming that there is no gap in the initial state
    u_grad_norm_ +=
        du_[t]
            .lpNorm<1>();  // assuming that there is no gap in the initial state

    lag_mul_[t + 1] = -qp_.results.y.segment(t * nx, nx);
    lag_mul_[t + 1] = -qp_.results.y.segment(t * nx, nx);
    y_[t] = qp_.results.z.segment(nin_count, nc);
    nin_count += nc;
    index_u += nu;
  }
  x_grad_norm_ = x_grad_norm_ / (T + 1);
  u_grad_norm_ = u_grad_norm_ / T;

  int nc = problem_->get_terminalModel()->get_ng();
  y_.back() = qp_.results.z.segment(nin_count, nc);

  STOP_PROFILER("SolverPROXQP::computeDirection");
}

void SolverPROXQP::checkKKTConditions() {
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  KKT_ = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    fs_flat_.segment(t * ndx, ndx) = fs_[t + 1];
    if (t == 0) {
      KKT_ = std::max(KKT_, (d->Lu + d->Fu.transpose() * lag_mul_[t + 1] +
                             d->Gu.transpose() * y_[t])
                                .lpNorm<Eigen::Infinity>());
      continue;
    }
    KKT_ = std::max(KKT_, (d->Lx + d->Fx.transpose() * lag_mul_[t + 1] -
                           lag_mul_[t] + d->Gx.transpose() * y_[t])
                              .lpNorm<Eigen::Infinity>());
    KKT_ = std::max(KKT_, (d->Lu + d->Fu.transpose() * lag_mul_[t + 1] +
                           d->Gu.transpose() * y_[t])
                              .lpNorm<Eigen::Infinity>());
  }
  const std::shared_ptr<ActionDataAbstract>& d_ter =
      problem_->get_terminalData();
  KKT_ = std::max(
      KKT_, (d_ter->Lx - lag_mul_.back() + d_ter->Gx.transpose() * y_.back())
                .lpNorm<Eigen::Infinity>());

  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, constraint_norm_);
}

double SolverPROXQP::tryStep(const double steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }

  START_PROFILER("SolverPROXQP::tryStep");
  cost_try_ = 0.;
  merit_try_ = 0;
  gap_norm_try_ = 0;
  constraint_norm_try_ = 0;

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    m->get_state()->integrate(xs_[t], steplength * dx_[t], xs_try_[t]);
    const std::size_t nu = m->get_nu();

    if (nu != 0) {
      us_try_[t].noalias() = us_[t] + steplength * du_[t];
    }
  }

  const std::shared_ptr<ActionModelAbstract>& m_ter =
      problem_->get_terminalModel();

  m_ter->get_state()->integrate(xs_.back(), steplength * dx_.back(),
                                xs_try_.back());

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];

    m->calc(d, xs_try_[t], us_try_[t]);
    cost_try_ += d->cost;
    m->get_state()->diff(xs_try_[t + 1], d->xnext, fs_try_[t + 1]);
    gap_norm_try_ += fs_try_[t + 1].lpNorm<1>();

    int nc = m->get_ng();
    auto lb = m->get_g_lb();
    auto ub = m->get_g_ub();
    constraint_norm_try_ +=
        (lb - d->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
    constraint_norm_try_ +=
        (d->g - ub).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverPROXQP::tryStep");
      throw_pretty("step_error");
    }
  }

  // Terminal state update
  const std::shared_ptr<ActionDataAbstract>& d_ter =
      problem_->get_terminalData();

  m_ter->calc(d_ter, xs_try_.back());
  cost_try_ += d_ter->cost;

  int nc = m_ter->get_ng();
  auto lb = m_ter->get_g_lb();
  auto ub = m_ter->get_g_ub();

  constraint_norm_try_ +=
      (lb - d_ter->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
  constraint_norm_try_ +=
      (d_ter->g - ub).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

  merit_try_ = cost_try_ + mu_dynamic_ * gap_norm_try_ +
               mu_constraint_ * constraint_norm_try_;

  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverPROXQP::tryStep");
    throw_pretty("step_error");
  }

  STOP_PROFILER("SolverPROXQP::tryStep");

  return merit_try_;
}

// const Eigen::MatrixXd& SolverPROXQP::get_P() const {return P_;};

void SolverPROXQP::printCallbacks() {
  if (this->get_iter() % 10 == 0) {
    std::cout << "iter     merit        cost         grad       step     "
                 "||gaps||       KKT       Constraint Norms    QP Iters";
    std::cout << std::endl;
  }
  if (KKT_ < termination_tol_) {
    std::cout << std::setw(4) << "END" << "  ";
  } else {
    std::cout << std::setw(4) << this->get_iter() + 1 << "  ";
  }
  std::cout << std::scientific << std::setprecision(5) << this->get_merit()
            << "  ";
  std::cout << std::scientific << std::setprecision(5) << this->get_cost()
            << "  ";
  std::cout << this->get_xgrad_norm() + this->get_ugrad_norm() << "  ";
  if (KKT_ < termination_tol_) {
    std::cout << std::fixed << std::setprecision(4) << " ---- " << "  ";
  } else {
    std::cout << std::fixed << std::setprecision(4) << this->get_steplength()
              << "  ";
  }
  std::cout << std::scientific << std::setprecision(5) << this->get_gap_norm()
            << "  ";
  std::cout << std::scientific << std::setprecision(5) << KKT_ << "    ";
  std::cout << std::scientific << std::setprecision(5) << constraint_norm_
            << "         ";
  std::cout << std::scientific << std::setprecision(5) << qp_iters_;

  std::cout << std::endl;
  std::cout << std::flush;
}

void SolverPROXQP::setCallbacks(bool inCallbacks) {
  with_callbacks_ = inCallbacks;
}

bool SolverPROXQP::getCallbacks() { return with_callbacks_; }

}  // namespace mim_solvers
