///////////////////////////////////////////////////////////////////////////////
//
// This file is a modified version of SolverKKT from the Crocoddyl library
// This modified version is used for testing purposes only
// Original file :
// https://github.com/loco-3d/crocoddyl/blob/devel/src/core/solvers/kkt.cpp
//
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mim_solvers/kkt.hpp"

namespace mim_solvers {

SolverKKT::SolverKKT(std::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverAbstract(problem),
      reg_incfactor_(10.),
      reg_decfactor_(10.),
      reg_min_(1e-9),
      reg_max_(1e9),
      cost_try_(0.),
      th_grad_(1e-12),
      was_feasible_(false) {
  allocateData();
  const std::size_t n_alphas = 10;
  preg_ = 0.;
  dreg_ = 0.;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., (double)n);
  }
  const std::size_t T = this->problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  fs_flat_.resize(ndx * (T + 1));
  fs_flat_.setZero();
}

SolverKKT::~SolverKKT() {}

bool SolverKKT::solve(const std::vector<Eigen::VectorXd>& init_xs,
                      const std::vector<Eigen::VectorXd>& init_us,
                      const std::size_t maxiter, const bool is_feasible,
                      const double) {
  setCandidate(init_xs, init_us, is_feasible);
  bool recalc = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    while (true) {
      try {
        computeDirection(recalc);
      } catch (std::exception& e) {
        recalc = false;
        if (preg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }

    expectedImprovement();

    // KKT termination criteria
    if (KKT_ <= termination_tol_) {
      return true;
    }

    for (std::vector<double>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
      steplength_ = *it;
      try {
        dV_ = tryStep(steplength_);
      } catch (std::exception& e) {
        continue;
      }
      dVexp_ = steplength_ * d_[0] + 0.5 * steplength_ * steplength_ * d_[1];
      if (d_[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_) {
        was_feasible_ = is_feasible_;
        setCandidate(xs_try_, us_try_, true);
        cost_ = cost_try_;
        break;
      }
    }
    stoppingCriteria();
    const std::size_t n_callbacks = callbacks_.size();
    if (n_callbacks != 0) {
      for (std::size_t c = 0; c < n_callbacks; ++c) {
        crocoddyl::CallbackAbstract& callback = *callbacks_[c];
        callback(*this);
      }
    }
    if (was_feasible_ && stop_ < th_stop_) {
      return true;
    }
  }
  return false;
}

void SolverKKT::computeDirection(const bool recalc) {
  const std::size_t T = problem_->get_T();
  if (recalc) {
    calcDiff();
  }
  computePrimalDual();
  const Eigen::VectorBlock<Eigen::VectorXd, Eigen::Dynamic> p_x =
      primal_.segment(0, ndx_);
  const Eigen::VectorBlock<Eigen::VectorXd, Eigen::Dynamic> p_u =
      primal_.segment(ndx_, nu_);

  std::size_t ix = 0;
  std::size_t iu = 0;
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();
    dxs_[t] = p_x.segment(ix, ndxi);
    dus_[t] = p_u.segment(iu, nui);
    lambdas_[t] = dual_.segment(ix, ndxi);
    ix += ndxi;
    iu += nui;
  }
  const std::size_t ndxi =
      problem_->get_terminalModel()->get_state()->get_ndx();
  dxs_.back() = p_x.segment(ix, ndxi);
  lambdas_.back() = dual_.segment(ix, ndxi);
  // KKT termination criteria
  checkKKTConditions();
}

double SolverKKT::tryStep(const double steplength) {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];

    m->get_state()->integrate(xs_[t], steplength * dxs_[t], xs_try_[t]);
    if (m->get_nu() != 0) {
      us_try_[t] = us_[t];
      us_try_[t] += steplength * dus_[t];
    }
  }
  const std::shared_ptr<crocoddyl::ActionModelAbstract> m =
      problem_->get_terminalModel();
  m->get_state()->integrate(xs_[T], steplength * dxs_[T], xs_try_[T]);
  cost_try_ = problem_->calc(xs_try_, us_try_);
  return cost_ - cost_try_;
}

void SolverKKT::checkKKTConditions() {
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    KKT_ = std::max(KKT_,
                    (d->Lx + d->Fx.transpose() * lambdas_[t + 1] - lambdas_[t])
                        .lpNorm<Eigen::Infinity>());
    KKT_ = std::max(KKT_, (d->Lu + d->Fu.transpose() * lambdas_[t + 1])
                              .lpNorm<Eigen::Infinity>());
    fs_flat_.segment(t * ndx, ndx) = fs_[t];
  }
  fs_flat_.tail(ndx) = fs_.back();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_ter =
      problem_->get_terminalData();
  KKT_ =
      std::max(KKT_, (d_ter->Lx - lambdas_.back()).lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
}

double SolverKKT::stoppingCriteria() {
  const std::size_t T = problem_->get_T();
  std::size_t ix = 0;
  std::size_t iu = 0;
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();

    dF.segment(ix, ndxi) = lambdas_[t];
    dF.segment(ix, ndxi).noalias() -= d->Fx.transpose() * lambdas_[t + 1];
    dF.segment(ndx_ + iu, nui).noalias() = -lambdas_[t + 1].transpose() * d->Fu;
    ix += ndxi;
    iu += nui;
  }
  const std::size_t ndxi =
      problem_->get_terminalModel()->get_state()->get_ndx();
  dF.segment(ix, ndxi) = lambdas_.back();
  stop_ = (kktref_.segment(0, ndx_ + nu_) + dF).squaredNorm() +
          kktref_.segment(ndx_ + nu_, ndx_).squaredNorm();
  return stop_;
}

const Eigen::Vector2d& SolverKKT::expectedImprovement() {
  d_ = Eigen::Vector2d::Zero();
  // -grad^T.primal
  d_(0) = -kktref_.segment(0, ndx_ + nu_).dot(primal_);
  // -(hessian.primal)^T.primal
  kkt_primal_.noalias() = kkt_.block(0, 0, ndx_ + nu_, ndx_ + nu_) * primal_;
  d_(1) = -kkt_primal_.dot(primal_);
  return d_;
}

const Eigen::MatrixXd& SolverKKT::get_kkt() const { return kkt_; }

const Eigen::VectorXd& SolverKKT::get_kktref() const { return kktref_; }

const Eigen::VectorXd& SolverKKT::get_primaldual() const { return primaldual_; }

const std::vector<Eigen::VectorXd>& SolverKKT::get_dxs() const { return dxs_; }

const std::vector<Eigen::VectorXd>& SolverKKT::get_dus() const { return dus_; }

const std::vector<Eigen::VectorXd>& SolverKKT::get_lambdas() const {
  return lambdas_;
}

std::size_t SolverKKT::get_nx() const { return nx_; }

std::size_t SolverKKT::get_ndx() const { return ndx_; }

std::size_t SolverKKT::get_nu() const { return nu_; }

double SolverKKT::calcDiff() {
  cost_ = problem_->calc(xs_, us_);
  cost_ = problem_->calcDiff(xs_, us_);

  // offset on constraint xnext = f(x,u) due to x0 = ref.
  const std::size_t cx0 =
      problem_->get_runningModels()[0]->get_state()->get_ndx();

  std::size_t ix = 0;
  std::size_t iu = 0;
  const std::size_t T = problem_->get_T();
  kkt_.block(ndx_ + nu_, 0, ndx_, ndx_).setIdentity();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m =
        problem_->get_runningModels()[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d =
        problem_->get_runningDatas()[t];
    const std::size_t ndxi = m->get_state()->get_ndx();
    const std::size_t nui = m->get_nu();

    // Computing the gap at the initial state
    if (t == 0) {
      m->get_state()->diff(problem_->get_x0(), xs_[0],
                           kktref_.segment(ndx_ + nu_, ndxi));
    }

    // Filling KKT matrix
    kkt_.block(ix, ix, ndxi, ndxi) = d->Lxx;
    kkt_.block(ix, ndx_ + iu, ndxi, nui) = d->Lxu;
    kkt_.block(ndx_ + iu, ix, nui, ndxi) = d->Lxu.transpose();
    kkt_.block(ndx_ + iu, ndx_ + iu, nui, nui) = d->Luu;
    kkt_.block(ndx_ + nu_ + cx0 + ix, ix, ndxi, ndxi) = -d->Fx;
    kkt_.block(ndx_ + nu_ + cx0 + ix, ndx_ + iu, ndxi, nui) = -d->Fu;

    // Filling KKT vector
    kktref_.segment(ix, ndxi) = d->Lx;
    kktref_.segment(ndx_ + iu, nui) = d->Lu;
    m->get_state()->diff(d->xnext, xs_[t + 1],
                         kktref_.segment(ndx_ + nu_ + cx0 + ix, ndxi));

    ix += ndxi;
    iu += nui;
  }
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& df =
      problem_->get_terminalData();
  const std::size_t ndxf =
      problem_->get_terminalModel()->get_state()->get_ndx();
  kkt_.block(ix, ix, ndxf, ndxf) = df->Lxx;
  kktref_.segment(ix, ndxf) = df->Lx;
  kkt_.block(0, ndx_ + nu_, ndx_ + nu_, ndx_) =
      kkt_.block(ndx_ + nu_, 0, ndx_, ndx_ + nu_).transpose();
  return cost_;
}

void SolverKKT::computePrimalDual() {
  primaldual_ = kkt_.lu().solve(-kktref_);
  primal_ = primaldual_.segment(0, ndx_ + nu_);
  dual_ = primaldual_.segment(ndx_ + nu_, ndx_);
}

void SolverKKT::increaseRegularization() {
  preg_ *= reg_incfactor_;
  if (preg_ > reg_max_) {
    preg_ = reg_max_;
  }
  dreg_ = preg_;
}

void SolverKKT::decreaseRegularization() {
  preg_ /= reg_decfactor_;
  if (preg_ < reg_min_) {
    preg_ = reg_min_;
  }
  dreg_ = preg_;
}

void SolverKKT::allocateData() {
  const std::size_t T = problem_->get_T();
  dxs_.resize(T + 1);
  dus_.resize(T);
  lambdas_.resize(T + 1);
  xs_try_.resize(T + 1);
  us_try_.resize(T);

  nx_ = 0;
  ndx_ = 0;
  nu_ = 0;
  const std::size_t nx = problem_->get_nx();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
    if (t == 0) {
      xs_try_[t] = problem_->get_x0();
    } else {
      xs_try_[t] = Eigen::VectorXd::Constant(nx, NAN);
    }
    const std::size_t nu = model->get_nu();
    us_try_[t] = Eigen::VectorXd::Constant(nu, NAN);
    dxs_[t] = Eigen::VectorXd::Zero(ndx);
    dus_[t] = Eigen::VectorXd::Zero(nu);
    lambdas_[t] = Eigen::VectorXd::Zero(ndx);
    nx_ += nx;
    ndx_ += ndx;
    nu_ += nu;
  }
  nx_ += nx;
  ndx_ += ndx;
  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();
  dxs_.back() = Eigen::VectorXd::Zero(ndx);
  lambdas_.back() = Eigen::VectorXd::Zero(ndx);

  // Set dimensions for kkt matrix and kkt_ref vector
  kkt_.resize(2 * ndx_ + nu_, 2 * ndx_ + nu_);
  kkt_.setZero();
  kktref_.resize(2 * ndx_ + nu_);
  kktref_.setZero();
  primaldual_.resize(2 * ndx_ + nu_);
  primaldual_.setZero();
  primal_.resize(ndx_ + nu_);
  primal_.setZero();
  kkt_primal_.resize(ndx_ + nu_);
  kkt_primal_.setZero();
  dual_.resize(ndx_);
  dual_.setZero();
  dF.resize(ndx_ + nu_);
  dF.setZero();
}

}  // namespace mim_solvers
