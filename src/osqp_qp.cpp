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


#include "mim_solvers/osqp_qp.hpp"
#include "mim_solvers/ddp.hpp"

using namespace crocoddyl;

namespace mim_solvers {

SolverOSQP_QP::SolverOSQP_QP(
    std::shared_ptr<crocoddyl::ShootingProblem> problem, SolverDDP* ddp)
    : problem_(problem), ddp_(ddp) {
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();

  // Allocate direction vectors
  dx_.resize(T + 1);
  du_.resize(T);
  dxtilde_.resize(T + 1);
  dutilde_.resize(T);

  // Allocate ADMM variables
  z_.resize(T + 1);
  z_relaxed_.resize(T + 1);
  z_prev_.resize(T + 1);
  y_.resize(T + 1);
  rho_vec_.resize(T + 1);
  inv_rho_vec_.resize(T + 1);
  rho_sparse_ = rho_sparse_base_;

  // Allocate temporary variables
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
    const std::size_t nc = model->get_ng();

    dx_[t].resize(ndx);
    dx_[t].setZero();
    du_[t].resize(nu);
    du_[t].setZero();
    dxtilde_[t].resize(ndx);
    dxtilde_[t].setZero();
    dutilde_[t].resize(nu);
    dutilde_[t].setZero();

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
    rho_vec_[t].setZero();
    inv_rho_vec_[t].resize(nc);
    inv_rho_vec_[t].setZero();
  }

  // Terminal state
  dx_.back().resize(ndx);
  dx_.back().setZero();
  dxtilde_.back().resize(ndx);
  dxtilde_.back().setZero();

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
  rho_vec_.back().setZero();
  inv_rho_vec_.back().resize(nc);
  inv_rho_vec_.back().setZero();
}

SolverOSQP_QP::~SolverOSQP_QP() {}

void SolverOSQP_QP::reset_params() {
  if (reset_rho_) {
    reset_rho_vec();
  }

  const std::size_t T = problem_->get_T();
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

void SolverOSQP_QP::set_rho_sparse(const double rho_sparse) {
  rho_sparse_ = rho_sparse;
  rho_sparse_base_ = rho_sparse;
  apply_rho_update(rho_sparse_);
}

void SolverOSQP_QP::reset_rho_vec() {
  rho_sparse_ = rho_sparse_base_;
  apply_rho_update(rho_sparse_);
}

void SolverOSQP_QP::apply_rho_update(const double rho_sparse_tmp) {
  START_PROFILER("SolverOSQP_QP::apply_rho_update");

  const std::size_t T = problem_->get_T();
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

  STOP_PROFILER("SolverOSQP_QP::apply_rho_update");
}

void SolverOSQP_QP::computeDirection() {
  START_PROFILER("SolverOSQP_QP::computeDirection");

  reset_params();

  if (equality_qp_initial_guess_) {
    backwardPass_without_constraints();
    forwardPass_without_constraints();
  }

  if (with_qp_callbacks_) {
    printQPCallbacks(0);
  }

  max_solve_time_reached_ = false;

  for (qp_iters_ = 1; qp_iters_ < max_qp_iters_ + 1; ++qp_iters_) {
    if (crocoddyl::getProfiler().take_time() - start_time_ >= max_solve_time_) {
      qp_iters_--;
      max_solve_time_reached_ = true;
      break;
    }

    if (qp_iters_ % rho_update_interval_ == 1 || rho_update_interval_ == 1) {
#ifdef CROCODDYL_WITH_MULTITHREADING
      if (problem_->get_nthreads() > 1)
        backwardPass_mt();
      else
#endif
        backwardPass();
    } else {
#ifdef CROCODDYL_WITH_MULTITHREADING
      if (problem_->get_nthreads() > 1)
        backwardPass_without_rho_update_mt();
      else
#endif
        backwardPass_without_rho_update();
    }

    forwardPass();
    update_lagrangian_parameters(qp_iters_);
    update_rho_vec(qp_iters_);

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

  STOP_PROFILER("SolverOSQP_QP::computeDirection");
}

void SolverOSQP_QP::update_rho_vec(const int iter) {
  const double scale = std::sqrt((norm_primal_ * norm_dual_rel_) /
                                 (norm_dual_ * norm_primal_rel_));
  rho_estimate_sparse_ =
      std::min(std::max(scale * rho_sparse_, rho_min_), rho_max_);

  if (iter % rho_update_interval_ == 0 && iter > 1) {
    if (rho_estimate_sparse_ > rho_sparse_ * adaptive_rho_tolerance_ ||
        rho_estimate_sparse_ < rho_sparse_ / adaptive_rho_tolerance_) {
      rho_sparse_ = rho_estimate_sparse_;
      apply_rho_update(rho_sparse_);
    }
  }
}

void SolverOSQP_QP::forwardPass() {
  auto profiler = crocoddyl::getProfiler().watcher("SolverOSQP_QP::forwardPass");
  profiler.start();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  const std::vector<Eigen::VectorXd>& k = ddp_->get_k();
  const auto& K = ddp_->K_;  // MatrixXdRowMajor
  const std::vector<Eigen::VectorXd>& fs = ddp_->get_fs();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    dutilde_[t] = -k[t];
    dutilde_[t].noalias() -= K[t] * dxtilde_[t];
    dxtilde_[t + 1] = fs[t + 1];
    dxtilde_[t + 1].noalias() += d->Fx * dxtilde_[t];
    dxtilde_[t + 1].noalias() += d->Fu * dutilde_[t];
  }
  profiler.stop();
}

void SolverOSQP_QP::forwardPass_without_constraints() {
  auto profiler = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::forwardPass_without_constraints");
  profiler.start();

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  const std::vector<Eigen::VectorXd>& k = ddp_->get_k();
  const auto& K = ddp_->K_;  // MatrixXdRowMajor
  const std::vector<Eigen::VectorXd>& fs = ddp_->get_fs();

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    du_[t] = -k[t];
    du_[t].noalias() -= K[t] * dx_[t];
    dx_[t + 1] = fs[t + 1];
    dx_[t + 1].noalias() += d->Fx * dx_[t];
    dx_[t + 1].noalias() += d->Fu * du_[t];
  }

  profiler.stop();
}

void SolverOSQP_QP::backwardPass() {
  static auto profiler_all =
      crocoddyl::getProfiler().watcher("SolverOSQP_QP::backwardPass");
  profiler_all.start();

  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  // Access DDP matrices via parent pointer
  std::vector<Eigen::MatrixXd>& Vxx = ddp_->Vxx_;
  std::vector<Eigen::VectorXd>& Vx = ddp_->Vx_;
  std::vector<Eigen::MatrixXd>& Qxx = ddp_->Qxx_;
  std::vector<Eigen::VectorXd>& Qx = ddp_->Qx_;
  std::vector<Eigen::MatrixXd>& Quu = ddp_->Quu_;
  std::vector<Eigen::VectorXd>& Qu = ddp_->Qu_;
  std::vector<Eigen::MatrixXd>& Qxu = ddp_->Qxu_;
  std::vector<Eigen::VectorXd>& k = ddp_->k_;
  auto& K = ddp_->K_;  // MatrixXdRowMajor
  auto& FuTVxx_p = ddp_->FuTVxx_p_;  // MatrixXdRowMajor
  MatrixXdRowMajor& FxTVxx_p = ddp_->FxTVxx_p_;
  Eigen::MatrixXd& Vxx_tmp = ddp_->Vxx_tmp_;
  std::vector<Eigen::LLT<Eigen::MatrixXd>>& Quu_llt = ddp_->Quu_llt_;
  const std::vector<Eigen::VectorXd>& fs = ddp_->get_fs();
  const double preg = ddp_->get_preg();
  const double dreg = ddp_->get_dreg();

  Vxx.back() = d_T->Lxx;
  Vxx.back().diagonal().array() += sigma_;
  Vx.back() = d_T->Lx;
  Vx.back().noalias() -= sigma_ * dx_.back();

  if (problem_->get_terminalModel()->get_ng()) {
    tmp_rhoGx_mat_.back().noalias() = rho_vec_.back().asDiagonal() * d_T->Gx;
    Vxx.back().noalias() += d_T->Gx.transpose() * tmp_rhoGx_mat_.back();
    tmp_dual_cwise_.back() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }
  if (!std::isnan(preg)) {
    Vxx.back().diagonal().array() += preg;
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx[t + 1];

    Vxx_fs_[t].noalias() = Vxx[t + 1] * fs[t + 1];
    tmp_Vx_ = Vxx_fs_[t] + Vx[t + 1];

    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();
    FxTVxx_p.noalias() = d->Fx.transpose() * Vxx_p;

    Qx[t] = d->Lx;
    Qx[t].noalias() -= sigma_ * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }
    Qx[t].noalias() += d->Fx.transpose() * tmp_Vx_;

    Qxx[t] = d->Lxx;
    Qxx[t].diagonal().array() += sigma_;
    if (t > 0 && nc != 0) {
      tmp_rhoGx_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gx;
      Qxx[t].noalias() += d->Gx.transpose() * tmp_rhoGx_mat_[t];
    }
    Qxx[t].noalias() += FxTVxx_p * d->Fx;

    if (nu != 0) {
      FuTVxx_p[t].noalias() = d->Fu.transpose() * Vxx_p;
      Qu[t] = d->Lu - sigma_ * du_[t];
      if (nc != 0) {
        Qu[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu[t].noalias() += d->Fu.transpose() * tmp_Vx_;

      Quu[t] = d->Luu;
      Quu[t].diagonal().array() += sigma_;
      Quu[t].noalias() += FuTVxx_p[t] * d->Fu;
      if (nc != 0) {
        tmp_rhoGu_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gu;
        Quu[t].noalias() += d->Gu.transpose() * tmp_rhoGu_mat_[t];
      }
      if (!std::isnan(dreg)) {
        Quu[t].diagonal().array() += dreg;
      }

      Qxu[t] = d->Lxu;
      if (t > 0 && nc != 0) {
        Qxu[t].noalias() += d->Gx.transpose() * tmp_rhoGu_mat_[t];
      }
      Qxu[t].noalias() += FxTVxx_p * d->Fu;
    }

    ddp_->computeGains(t);

    Vx[t] = Qx[t];
    Vxx[t] = Qxx[t];
    if (nu != 0) {
      Vx[t].noalias() -= K[t].transpose() * Qu[t];
      Vxx[t].noalias() -= Qxu[t] * K[t];
    }
    Vxx_tmp = 0.5 * (Vxx[t] + Vxx[t].transpose());
    Vxx[t] = Vxx_tmp;
    if (!std::isnan(preg)) {
      Vxx[t].diagonal().array() += preg;
    }
  }
  profiler_all.stop();
}

void SolverOSQP_QP::backwardPass_without_constraints() {
  static auto profiler_all = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::backwardPass_without_constraints");
  profiler_all.start();

  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  // Access DDP matrices
  std::vector<Eigen::MatrixXd>& Vxx = ddp_->Vxx_;
  std::vector<Eigen::VectorXd>& Vx = ddp_->Vx_;
  std::vector<Eigen::MatrixXd>& Qxx = ddp_->Qxx_;
  std::vector<Eigen::VectorXd>& Qx = ddp_->Qx_;
  std::vector<Eigen::MatrixXd>& Quu = ddp_->Quu_;
  std::vector<Eigen::VectorXd>& Qu = ddp_->Qu_;
  std::vector<Eigen::MatrixXd>& Qxu = ddp_->Qxu_;
  auto& FuTVxx_p = ddp_->FuTVxx_p_;  // MatrixXdRowMajor
  MatrixXdRowMajor& FxTVxx_p = ddp_->FxTVxx_p_;
  Eigen::MatrixXd& Vxx_tmp = ddp_->Vxx_tmp_;
  const std::vector<Eigen::VectorXd>& fs = ddp_->get_fs();
  const double preg = ddp_->get_preg();
  const double dreg = ddp_->get_dreg();

  Vxx.back() = d_T->Lxx;
  Vx.back() = d_T->Lx;

  if (!std::isnan(preg)) {
    Vxx.back().diagonal().array() += preg;
  }

  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx[t + 1];
    tmp_Vx_.noalias() = Vxx[t + 1] * fs[t + 1];
    tmp_Vx_ += Vx[t + 1];

    const std::size_t nu = m->get_nu();
    FxTVxx_p.noalias() = d->Fx.transpose() * Vxx_p;

    Qx[t] = d->Lx;
    Qx[t].noalias() += d->Fx.transpose() * tmp_Vx_;

    Qxx[t] = d->Lxx;
    Qxx[t].noalias() += FxTVxx_p * d->Fx;

    if (nu != 0) {
      FuTVxx_p[t].noalias() = d->Fu.transpose() * Vxx_p;
      Qu[t] = d->Lu;
      Qu[t].noalias() += d->Fu.transpose() * tmp_Vx_;

      Quu[t] = d->Luu;
      Quu[t].noalias() += FuTVxx_p[t] * d->Fu;

      Qxu[t] = d->Lxu;
      Qxu[t].noalias() += FxTVxx_p * d->Fu;

      if (!std::isnan(dreg)) {
        Quu[t].diagonal().array() += dreg;
      }
    }

    ddp_->computeGains(t);

    Vx[t] = Qx[t];
    Vxx[t] = Qxx[t];
    if (nu != 0) {
      Vx[t].noalias() -= ddp_->K_[t].transpose() * Qu[t];
      Vxx[t].noalias() -= Qxu[t] * ddp_->K_[t];
    }
    Vxx_tmp = 0.5 * (Vxx[t] + Vxx[t].transpose());
    Vxx[t] = Vxx_tmp;

    if (!std::isnan(preg)) {
      Vxx[t].diagonal().array() += preg;
    }
  }
  profiler_all.stop();
}

void SolverOSQP_QP::backwardPass_without_rho_update() {
  static auto profiler_all = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::backwardPass_without_rho_update");
  profiler_all.start();

  const std::shared_ptr<crocoddyl::ActionModelAbstract>& m_T =
      problem_->get_terminalModel();
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  std::vector<Eigen::VectorXd>& Vx = ddp_->Vx_;
  std::vector<Eigen::VectorXd>& Qx = ddp_->Qx_;
  std::vector<Eigen::VectorXd>& Qu = ddp_->Qu_;
  std::vector<Eigen::VectorXd>& k = ddp_->k_;
  auto& K = ddp_->K_;  // MatrixXdRowMajor
  std::vector<Eigen::LLT<Eigen::MatrixXd>>& Quu_llt = ddp_->Quu_llt_;
  const std::vector<Eigen::VectorXd>& fs = ddp_->get_fs();

  Vx.back() = d_T->Lx;
  Vx.back().noalias() -= sigma_ * dx_.back();

  if (m_T->get_ng()) {
    tmp_dual_cwise_.back() = y_.back();
    tmp_dual_cwise_.back().noalias() -= rho_vec_.back().cwiseProduct(z_.back());
    Vx.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }

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

    tmp_Vx_ = Vxx_fs_[t] + Vx[t + 1];
    Qx[t] = d->Lx;
    Qx[t].noalias() -= sigma_ * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }
    Qx[t].noalias() += d->Fx.transpose() * tmp_Vx_;

    if (nu != 0) {
      Qu[t] = d->Lu;
      Qu[t].noalias() -= sigma_ * du_[t];
      if (nc != 0) {
        Qu[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu[t].noalias() += d->Fu.transpose() * tmp_Vx_;
    }

    k[t] = Qu[t];
    Quu_llt[t].solveInPlace(k[t]);

    Vx[t] = Qx[t];
    if (nu != 0) {
      Vx[t].noalias() -= K[t].transpose() * Qu[t];
    }
  }
  profiler_all.stop();
}

#ifdef CROCODDYL_WITH_MULTITHREADING
void SolverOSQP_QP::backwardPass_mt() {
  static auto profiler_all =
      crocoddyl::getProfiler().watcher("SolverOSQP_QP::backwardPass_mt");
  static auto profiler_lock =
      crocoddyl::getProfiler().watcher("SolverOSQP_QP::backwardPass_mt::lock");
  profiler_all.start();

  const std::shared_ptr<crocoddyl::ActionDataAbstract>& d_T =
      problem_->get_terminalData();

  std::vector<Eigen::MatrixXd>& Vxx = ddp_->Vxx_;
  std::vector<Eigen::VectorXd>& Vx = ddp_->Vx_;
  std::vector<Eigen::MatrixXd>& Qxx = ddp_->Qxx_;
  std::vector<Eigen::VectorXd>& Qx = ddp_->Qx_;
  std::vector<Eigen::MatrixXd>& Quu = ddp_->Quu_;
  std::vector<Eigen::VectorXd>& Qu = ddp_->Qu_;
  std::vector<Eigen::MatrixXd>& Qxu = ddp_->Qxu_;
  auto& FuTVxx_p = ddp_->FuTVxx_p_;  // MatrixXdRowMajor
  MatrixXdRowMajor& FxTVxx_p = ddp_->FxTVxx_p_;
  Eigen::MatrixXd& Vxx_tmp = ddp_->Vxx_tmp_;
  const std::vector<Eigen::VectorXd>& fs = ddp_->get_fs();
  const double preg = ddp_->get_preg();
  const double dreg = ddp_->get_dreg();

  Vxx.back() = d_T->Lxx;
  Vxx.back().diagonal().array() += sigma_;
  Vx.back() = d_T->Lx;
  Vx.back().noalias() -= sigma_ * dx_.back();

  if (problem_->get_terminalModel()->get_ng()) {
    tmp_rhoGx_mat_.back().noalias() = rho_vec_.back().asDiagonal() * d_T->Gx;
    Vxx.back().noalias() += d_T->Gx.transpose() * tmp_rhoGx_mat_.back();
    tmp_dual_cwise_.back() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }
  if (!std::isnan(preg)) {
    Vxx.back().diagonal().array() += preg;
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

    Qx[t] = d->Lx;
    Qx[t].noalias() -= sigma_ * dx_[t];
    if (nc != 0) {
      if (t > 0 || nu != 0) {
        tmp_dual_cwise_[t] = y_[t];
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      }
      if (t > 0) {
        Qx[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
      }
    }

    Qxx[t] = d->Lxx;
    Qxx[t].diagonal().array() += sigma_;
    if (t > 0 && nc != 0) {
      tmp_rhoGx_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gx;
      Qxx[t].noalias() += d->Gx.transpose() * tmp_rhoGx_mat_[t];
    }

    if (nu != 0) {
      Qu[t] = d->Lu - sigma_ * du_[t];
      if (nc != 0) {
        Qu[t] += d->Gu.transpose() * tmp_dual_cwise_[t];
      }

      Quu[t] = d->Luu;
      Quu[t].diagonal().array() += sigma_;
      if (nc != 0) {
        tmp_rhoGu_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gu;
        Quu[t].noalias() += d->Gu.transpose() * tmp_rhoGu_mat_[t];
      }
      if (!std::isnan(dreg)) {
        Quu[t].diagonal().array() += dreg;
      }

      Qxu[t] = d->Lxu;
      if (t > 0 && nc != 0) {
        Qxu[t].noalias() += d->Gx.transpose() * tmp_rhoGu_mat_[t];
      }
    }
  }

  profiler_lock.start();
  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();

    const Eigen::MatrixXd& Vxx_p = Vxx[t + 1];
    FxTVxx_p.noalias() = d->Fx.transpose() * Vxx_p;
    Qxx[t].noalias() += FxTVxx_p * d->Fx;

    Vxx_fs_[t].noalias() = Vxx[t + 1] * fs[t + 1];
    tmp_Vx_ = Vxx_fs_[t] + Vx[t + 1];
    Qx[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    if (nu != 0) {
      FuTVxx_p[0].noalias() = d->Fu.transpose() * Vxx_p;
      Quu[t].noalias() += FuTVxx_p[0] * d->Fu;
      Qu[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      Qxu[t].noalias() += FxTVxx_p * d->Fu;
    }

    ddp_->computeGains(t);
    Vx[t] = Qx[t];
    Vxx[t] = Qxx[t];
    if (nu != 0) {
      Vx[t].noalias() -= ddp_->K_[t].transpose() * Qu[t];
      Vxx[t].noalias() -= Qxu[t] * ddp_->K_[t];
    }
    Vxx_tmp.triangularView<Eigen::Upper>() =
        (0.5 * (Vxx[t] + Vxx[t].transpose())).triangularView<Eigen::Upper>();
    Vxx[t] = Vxx_tmp.selfadjointView<Eigen::Upper>();

    if (!std::isnan(preg)) {
      Vxx[t].diagonal().array() += preg;
    }
  }
  profiler_lock.stop();
  profiler_all.stop();
}

void SolverOSQP_QP::backwardPass_without_rho_update_mt() {
  static auto profiler_all = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::backwardPass_wo_rho_update_mt");
  static auto profiler_mt1 = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::backwardPass_wo_rho_update_mt::init_Qx_Qu");
  static auto profiler_pass = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::backwardPass_wo_rho_update_mt::update_Qx_Qu_Vx");
  static auto profiler_mt2 = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::backwardPass_wo_rho_update_mt::k");

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

  std::vector<Eigen::VectorXd>& Vx = ddp_->Vx_;
  std::vector<Eigen::VectorXd>& Qx = ddp_->Qx_;
  std::vector<Eigen::VectorXd>& Qu = ddp_->Qu_;
  std::vector<Eigen::VectorXd>& k = ddp_->k_;
  auto& K = ddp_->K_;  // MatrixXdRowMajor
  std::vector<Eigen::LLT<Eigen::MatrixXd>>& Quu_llt = ddp_->Quu_llt_;

  Vx.back().noalias() = d_T->Lx - sigma_ * dx_.back();

  if (m_T->get_ng()) {
    tmp_dual_cwise_.back().noalias() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }

  profiler_mt1.start();
#pragma omp parallel for num_threads(problem_->get_nthreads())
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();
    const std::size_t nc = m->get_ng();

    Qx[t].noalias() = d->Lx - sigma_ * dx_[t];
    if (nc != 0 && (t > 0 || nu != 0)) {
      tmp_dual_cwise_[t].noalias() = y_[t] - rho_vec_[t].cwiseProduct(z_[t]);
    }
    if (nc != 0 && t > 0) {
      Qx[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
    }

    if (nu != 0) {
      Qu[t].noalias() = d->Lu - sigma_ * du_[t];
      if (nc != 0) {
        Qu[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
    }
  }
  profiler_mt1.stop();

  profiler_pass.start();
  for (int t = static_cast<int>(T - 1); t >= 0; --t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();

    tmp_Vx_ = Vxx_fs_[t] + Vx[t + 1];
    Qx[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    Vx[t] = Qx[t];

    if (nu != 0) {
      Qu[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      Vx[t].noalias() -= K[t].transpose() * Qu[t];
    }
  }
  profiler_pass.stop();

  profiler_mt2.start();
#pragma omp parallel for num_threads(problem_->get_nthreads())
  for (std::size_t t = 0; t < T; ++t) {
    k[t] = Qu[t];
    Quu_llt[t].solveInPlace(k[t]);
  }
  profiler_mt2.stop();
  profiler_all.stop();
}
#endif  // CROCODDYL_WITH_MULTITHREADING

void SolverOSQP_QP::update_lagrangian_parameters(const int iter) {
  static auto profiler_all = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::update_lagrangian_parameters");
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
#endif
  for (std::size_t t = 0; t < T; ++t) {
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

void SolverOSQP_QP::printQPCallbacks(const int iter) {
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

void SolverOSQP_QP::setQPCallbacks(const bool inQPCallbacks) {
  with_qp_callbacks_ = inQPCallbacks;
}

}  // namespace mim_solvers
