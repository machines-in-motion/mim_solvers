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
#include <crocoddyl/core/utils/exception.hpp>


#include "mim_solvers/osqp_qp.hpp"

using namespace crocoddyl;

namespace mim_solvers {

SolverOSQP_QP::SolverOSQP_QP(
    std::shared_ptr<crocoddyl::ShootingProblem> problem)
    : problem_(problem) {
  allocateData();
}

void SolverOSQP_QP::allocateData() {
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
  tmp_vec_u_.resize(T);
  Vxx_fs_.resize(T);
  fs_.resize(T + 1);

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
    fs_[t].resize(ndx);
    fs_[t].setZero();

    rho_vec_[t].resize(nc);
    rho_vec_[t].setZero();
    inv_rho_vec_[t].resize(nc);
    inv_rho_vec_[t].setZero();
  }

  // ========================
  // Allocate DDP data (Vxx_, Qxx_, K_, k, etc.)
  // ========================
  Vxx_.resize(T + 1);
  Vx_.resize(T + 1);
  Qxx_.resize(T);
  Qxu_.resize(T);
  Quu_.resize(T);
  Qx_.resize(T);
  Qu_.resize(T);
  K_.resize(T);
  k_.resize(T);

  FuTVxx_p_.resize(T);
  Quu_llt_.resize(T);
  Quuk_.resize(T);

  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Vxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
    Vx_[t] = Eigen::VectorXd::Zero(ndx);
    Qxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
    Qxu_[t] = Eigen::MatrixXd::Zero(ndx, nu);
    Quu_[t] = Eigen::MatrixXd::Zero(nu, nu);
    Qx_[t] = Eigen::VectorXd::Zero(ndx);
    Qu_[t] = Eigen::VectorXd::Zero(nu);
    K_[t] = MatrixXdRowMajor::Zero(nu, ndx);
    k_[t] = Eigen::VectorXd::Zero(nu);

    FuTVxx_p_[t] = MatrixXdRowMajor::Zero(nu, ndx);
    Quu_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nu);
    Quuk_[t] = Eigen::VectorXd(nu);
  }
  Vxx_.back() = Eigen::MatrixXd::Zero(ndx, ndx);
  Vxx_tmp_ = Eigen::MatrixXd::Zero(ndx, ndx);
  Vx_.back() = Eigen::VectorXd::Zero(ndx);

  FxTVxx_p_ = MatrixXdRowMajor::Zero(ndx, ndx);
  fTVxx_p_ = Eigen::VectorXd::Zero(ndx);

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
  fs_.back().resize(ndx);
  fs_.back().setZero();
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
#endif // CROCODDYL_WITH_MULTITHREADING
        backwardPass();
    } else {
#ifdef CROCODDYL_WITH_MULTITHREADING
      if (problem_->get_nthreads() > 1)
        backwardPass_without_rho_update_mt();
      else
#endif // CROCODDYL_WITH_MULTITHREADING
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

  STOP_PROFILER("SolverOSQP_QP::computeDirection");
}

void SolverOSQP_QP::update_rho_vec(const int iter) {
  const double scale = std::sqrt((norm_primal_ * norm_dual_rel_) /
                                 (norm_dual_ * norm_primal_rel_));
  rho_estimate_sparse_ =
      std::min(std::max(scale * rho_sparse_, rho_min_), rho_max_);

  if (iter % rho_update_interval_ == 0) { //&& iter > 1) {
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

void SolverOSQP_QP::forwardPass_without_constraints() {
  auto profiler = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::forwardPass_without_constraints");
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

void SolverOSQP_QP::backwardPass() {
  static auto profiler_all =
      crocoddyl::getProfiler().watcher("SolverOSQP_QP::backwardPass");
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

    Qxx_[t] = d->Lxx;
    Qxx_[t].diagonal().array() += sigma_;
    if (t > 0 && nc != 0) {
      tmp_rhoGx_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gx;
      Qxx_[t].noalias() += d->Gx.transpose() * tmp_rhoGx_mat_[t];
    }
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;

    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      Qu_[t] = d->Lu - sigma_ * du_[t];
      if (nc != 0) {
        Qu_[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;

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

      Qxu_[t] = d->Lxu;
      if (t > 0 && nc != 0) {
        Qxu_[t].noalias() += d->Gx.transpose() * tmp_rhoGu_mat_[t];
      }
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
    }

    computeGains(t);

    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;
    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
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

    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= sigma_ * dxtilde_[t];
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;

    Qxx_[t] = d->Lxx;
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;

    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      Qu_[t] = d->Lu;
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;

      Quu_[t] = d->Luu;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;

      Qxu_[t] = d->Lxu;
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;

      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
      }
    }

    computeGains(t);

    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;

    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
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

  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= sigma_ * dx_.back();

  if (m_T->get_ng()) {
    tmp_dual_cwise_.back() = y_.back();
    tmp_dual_cwise_.back().noalias() -= rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
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

    if (nu != 0) {
      Qu_[t] = d->Lu;
      Qu_[t].noalias() -= sigma_ * du_[t];
      if (nc != 0) {
        Qu_[t].noalias() += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
    }

    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);

    Vx_[t] = Qx_[t];
    if (nu != 0) {
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
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
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    }
    Vxx_tmp_.triangularView<Eigen::Upper>() =
        (0.5 * (Vxx_[t] + Vxx_[t].transpose())).triangularView<Eigen::Upper>();
    Vxx_[t] = Vxx_tmp_.selfadjointView<Eigen::Upper>();

    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
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

  Vx_.back().noalias() = d_T->Lx - sigma_ * dx_.back();

  if (m_T->get_ng()) {
    tmp_dual_cwise_.back().noalias() =
        y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }

  profiler_mt1.start();
#pragma omp parallel for num_threads(problem_->get_nthreads())
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
#pragma omp parallel for num_threads(problem_->get_nthreads())
  for (std::size_t t = 0; t < T; ++t) {
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
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

void SolverOSQP_QP::computeGains(const std::size_t t) {
  static auto profiler_all =
      crocoddyl::getProfiler().watcher("SolverOSQP_QP::computeGains");
  static auto profiler_Quu_inv =
      crocoddyl::getProfiler().watcher("SolverOSQP_QP::computeGains::Quu_inv");
  static auto profiler_Quu_inv_Qux = crocoddyl::getProfiler().watcher(
      "SolverOSQP_QP::computeGains::Quu_inv_Qux");
  profiler_all.start();

  const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
  if (nu > 0) {
    profiler_Quu_inv.start();
    Quu_llt_[t].compute(Quu_[t]);
    profiler_Quu_inv.stop();
    const Eigen::ComputationInfo& info = Quu_llt_[t].info();
    if (info != Eigen::Success) {
      profiler_all.stop();
      throw_pretty("backward_error");
    }
    K_[t] = Qxu_[t].transpose();

    profiler_Quu_inv_Qux.start();
    Quu_llt_[t].solveInPlace(K_[t]);
    profiler_Quu_inv_Qux.stop();
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
  }
  profiler_all.stop();
}

void SolverOSQP_QP::increaseRegularization() {
  preg_ *= reg_incfactor_;
  if (preg_ > reg_max_) {
    preg_ = reg_max_;
  }
  dreg_ = preg_;
}

}  // namespace mim_solvers
