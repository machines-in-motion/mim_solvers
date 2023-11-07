///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif  // CROCODDYL_WITH_MULTITHREADING

#include <iostream>
#include <iomanip>

#include <crocoddyl/core/utils/exception.hpp>
#include "mim_solvers/csqp.hpp"

using namespace crocoddyl;

namespace mim_solvers {

// const std::vector<boost::shared_ptr<ConstraintModelAbstract> >& constraint_models
SolverCSQP::SolverCSQP(boost::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverDDP(problem){
      
      const std::size_t T = this->problem_->get_T();
      const std::size_t ndx = problem_->get_ndx();
      sigma_diag_x = sigma_* Eigen::MatrixXd::Identity(ndx, ndx);
      sigma_diag_u.resize(T);
      Cdx_Cdu.resize(T+1);
      constraint_list_.resize(filter_size_);
      gap_list_.resize(filter_size_);
      cost_list_.resize(filter_size_);
      fs_try_.resize(T + 1);
      fs_flat_.resize(ndx*(T + 1));
      fs_flat_.setZero();
      lag_mul_.resize(T+1);
      du_.resize(T);
      
      dx_.resize(T+1); dxtilde_.resize(T+1);
      du_.resize(T); dutilde_.resize(T);
      z_.resize(T+1); z_relaxed_.resize(T+1); y_.resize(T+1); rho_vec_.resize(T+1); inv_rho_vec_.resize(T+1);
      z_prev_.resize(T+1);
      xs_try_.resize(T+1); us_try_.resize(T);

      rho_sparse_base_ = rho_sparse_;

      const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
      for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
        const std::size_t nu = model->get_nu();
        sigma_diag_u[t] = sigma_ * Eigen::MatrixXd::Identity(nu, nu);
        xs_try_[t] = model->get_state()->zero();
        us_try_[t] = Eigen::VectorXd::Zero(nu);

        dx_[t].resize(ndx); du_[t].resize(nu);
        dxtilde_[t].resize(ndx); dutilde_[t].resize(nu);
        fs_try_[t].resize(ndx);
        lag_mul_[t].resize(ndx); 
        lag_mul_[t].setZero();
        dx_[t].setZero(); dxtilde_[t].setZero();
        du_[t] = Eigen::VectorXd::Zero(nu);
        dutilde_[t] = Eigen::VectorXd::Zero(nu);

        fs_try_[t] = Eigen::VectorXd::Zero(ndx);

        std::size_t nc = model->get_ng(); 
        Cdx_Cdu[t].resize(nc); Cdx_Cdu[t].setZero();


        z_[t].resize(nc); z_[t].setZero();
        z_prev_[t].resize(nc); z_prev_[t].setZero();
        z_relaxed_[t].resize(nc); z_relaxed_[t].setZero();
        y_[t].resize(nc); y_[t].setZero();
        rho_vec_[t].resize(nc); 
        inv_rho_vec_[t].resize(nc); 

        auto lb = model->get_g_lb(); 
        auto ub = model->get_g_ub();
        double infty = std::numeric_limits<double>::infinity();

        for (std::size_t k = 0; k < nc; ++k){
          if (lb[k] == -infty && ub[k] == infty){
              rho_vec_[t][k] = rho_min_;
              inv_rho_vec_[t][k] = 1.0/rho_min_;
          }
          else if (abs(lb[k] - ub[k]) <= 1e-6){
              rho_vec_[t][k] = 1e3 * rho_sparse_;
              inv_rho_vec_[t][k] = 1/(1e3 * rho_sparse_);

          }
          else if (lb[k] < ub[k]){
              rho_vec_[t][k] = rho_sparse_;
              inv_rho_vec_[t][k] = rho_sparse_;

          }
        }
      }

      xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();

      lag_mul_.back().resize(ndx);
      lag_mul_.back().setZero();
      dx_.back().resize(ndx); dxtilde_.back().resize(ndx);
      dx_.back().setZero(); dxtilde_.back().setZero();
      fs_try_.back().resize(ndx);
      fs_try_.back() = Eigen::VectorXd::Zero(ndx);

      // Constraint Models
      std::size_t nc = problem_->get_terminalModel()->get_ng(); 
      z_.back().resize(nc); z_.back().setZero();
      z_prev_.back().resize(nc); z_prev_.back().setZero();
      z_relaxed_.back().resize(nc); z_relaxed_.back().setZero();
      y_.back().resize(nc); y_.back().setZero();
      rho_vec_.back().resize(nc);
      inv_rho_vec_.back().resize(nc);

      for (std::size_t k = 0; k < nc; ++k){
        auto lb = problem_->get_terminalModel()->get_g_lb(); 
        auto ub = problem_->get_terminalModel()->get_g_ub();
        double infty = std::numeric_limits<double>::infinity();
        if (lb[k] == -infty && ub[k] == infty){
            rho_vec_.back()[k] = rho_min_;
            inv_rho_vec_.back()[k] = 1/rho_min_;
        }
        else if (abs(lb[k] - ub[k]) <= 1e-6){
            rho_vec_.back()[k] = 1e3 * rho_sparse_;
            inv_rho_vec_.back()[k] = 1/(1e3 * rho_sparse_);
        }
        else if (lb[k] < ub[k]){
            rho_vec_.back()[k] = rho_sparse_;
            inv_rho_vec_.back()[k] = 1/rho_sparse_;
        }
      }

      const std::size_t n_alphas = 10;
      alphas_.resize(n_alphas);
      for (std::size_t n = 0; n < n_alphas; ++n) {
        alphas_[n] = 1. / pow(2., static_cast<double>(n));
      }
      if (th_stepinc_ < alphas_[n_alphas - 1]) {
        th_stepinc_ = alphas_[n_alphas - 1];
        std::cerr << "Warning: th_stepinc has higher value than lowest alpha value, set to "
                  << std::to_string(alphas_[n_alphas - 1]) << std::endl;
      }
    }



void SolverCSQP::reset_params(){

  rho_sparse_ = rho_sparse_base_;
  
  const std::size_t T = this->problem_->get_T();
  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
      std::size_t nc = model->get_ng();
      z_[t].setZero();
      z_prev_[t].setZero();
      z_relaxed_[t].setZero();
      
      // Warm start y
      if(!warm_start_y_){
        y_[t].setZero();
      } 
        
      auto lb = model->get_g_lb(); 
      auto ub = model->get_g_ub();
      double infty = std::numeric_limits<double>::infinity();

      // Reset rho
      if(reset_rho_){
        for (std::size_t k = 0; k < nc; ++k){
          if (lb[k] == -infty && ub[k] == infty){
              rho_vec_[t][k] = rho_min_;
              inv_rho_vec_[t][k] = 1/rho_min_;

          }
          else if (abs(lb[k] - ub[k]) <= 1e-6){
              rho_vec_[t][k] = 1e3 * rho_sparse_;
              inv_rho_vec_[t][k] = 1.0/(1e3 * rho_sparse_);
          }
          else if (lb[k] < ub[k]){
              rho_vec_[t][k] = rho_sparse_;
              inv_rho_vec_[t][k] = 1/rho_sparse_;

          }
        }
      }

    }
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem_->get_terminalModel();
    lag_mul_.back().setZero();
    std::size_t nc = model->get_ng();
    z_.back().setZero();
    z_prev_.back().setZero();
    z_relaxed_.back().setZero();
    // Warm-start y
    if(!warm_start_y_){
      y_.back().setZero();
    }
    // Reset rho
    if(reset_rho_){
      for (std::size_t k = 0; k < nc; ++k){
        auto lb = problem_->get_terminalModel()->get_g_lb(); 
        auto ub = problem_->get_terminalModel()->get_g_ub();
        double infty = std::numeric_limits<double>::infinity();
        if (lb[k] == -infty && ub[k] == infty){
            rho_vec_.back()[k] = rho_min_;
            inv_rho_vec_.back()[k] = 1/rho_min_;
        }
        else if (abs(lb[k] - ub[k]) <= 1e-6){
            rho_vec_.back()[k] = 1e3 * rho_sparse_;
            inv_rho_vec_.back()[k] = 1/(1e3 * rho_sparse_);
        }
        else if (lb[k] < ub[k]){
            rho_vec_.back()[k] = rho_sparse_;
            inv_rho_vec_.back()[k] = 1/rho_sparse_;
        }
      }
    }
}

SolverCSQP::~SolverCSQP() {}

bool SolverCSQP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                       const std::size_t maxiter, const bool is_feasible, const double reginit) {
  
  (void)is_feasible;
  (void)reginit;

  START_PROFILER("SolverCSQP::solve");
  if (problem_->is_updated()) {
    resizeData();
  }
  xs_try_[0] = problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible
  setCandidate(init_xs, init_us, false);

  // REMOVED REGULARIZATION : 
  // if (std::isnan(reginit)) {
  //   xreg_ = reg_min_;
  //   ureg_ = reg_min_;
  // } else {
  //   xreg_ = reginit;
  //   ureg_ = reginit;
  // }

  for (iter_ = 0; iter_ < maxiter; ++iter_) {


    was_feasible_ = false;
    bool recalcDiff = true;
    // computeDirection(recalcDiff);

    while (true) {
      try {
        computeDirection(recalcDiff);
      } 
      catch (std::exception& e) {
        recalcDiff = false;
        increaseRegularization();
        if (xreg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }

    // KKT termination criteria
    if(use_kkt_criteria_){
      if (KKT_  <= termination_tol_) {
        STOP_PROFILER("SolverCSQP::solve");
        return true;
      }
    }  

    constraint_list_.push_back(constraint_norm_);
    gap_list_.push_back(gap_norm_);
    cost_list_.push_back(cost_);

    // We need to recalculate the derivatives when the step length passes
    for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
      steplength_ = *it;
      try {
        merit_try_ = tryStep(steplength_);
      } catch (std::exception& e) {
        continue;
      }
      // Filter line search criteria 
      // Equivalent to heuristic cost_ > cost_try_ || gap_norm_ > gap_norm_try_ when filter_size=1
      if(use_filter_line_search_){
        is_worse_than_memory_ = false;
        std::size_t count = 0.; 
        while( count < filter_size_ && is_worse_than_memory_ == false and count <= iter_){
          is_worse_than_memory_ = cost_list_[filter_size_-1-count] <= cost_try_ && gap_list_[filter_size_-1-count] <= gap_norm_try_ && constraint_list_[filter_size_-1-count] <= constraint_norm_try_;
          count++;
        }
        if( is_worse_than_memory_ == false ) {
          setCandidate(xs_try_, us_try_, false);
          recalcDiff = true;
          break;
        } 
      }
      // Line-search criteria using merit function
      else{
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
      if (xreg_ == reg_max_) {
        STOP_PROFILER("SolverCSQP::solve");
        return false;
      }
    }

    if(with_callbacks_){
      printCallbacks();
    }
  }
  STOP_PROFILER("SolverCSQP::solve");
  return false;
}


void SolverCSQP::calc(const bool recalc){
  if (recalc){
    problem_->calc(xs_, us_);
    cost_ = problem_->calcDiff(xs_, us_);
  }

  gap_norm_ = 0;
  constraint_norm_ = 0;
  // double infty = std::numeric_limits<double>::infinity();

  const std::size_t T = problem_->get_T();
  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();

  for (std::size_t t = 0; t < T; ++t) {

    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    m->get_state()->diff(xs_[t + 1], d->xnext, fs_[t + 1]);

    gap_norm_ += fs_[t+1].lpNorm<1>();  

    std::size_t nc = m->get_ng();
    auto lb = m->get_g_lb(); 
    auto ub = m->get_g_ub();
    constraint_norm_ += (lb - d->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
    constraint_norm_ += (d->g - ub).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

  }

  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_T = problem_->get_terminalData();
  std::size_t nc = problem_->get_terminalModel()->get_ng();
  auto lb = problem_->get_terminalModel()->get_g_lb();
  auto ub = problem_->get_terminalModel()->get_g_ub();

  constraint_norm_ += (lb - d_T->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
  constraint_norm_ += (d_T->g - ub).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

  merit_ = cost_ + mu_*gap_norm_ + mu2_*constraint_norm_;

}


void SolverCSQP::computeDirection(const bool recalcDiff){
  START_PROFILER("SolverCSQP::computeDirection");
  if (recalcDiff) {
    calc(recalcDiff);
  }
  if(use_kkt_criteria_){
    checkKKTConditions();
  }

  if (warm_start_){
    reset_params();
    backwardPass_without_constraints();
    forwardPass();
    update_lagrangian_parameters(false);
  }
  bool converged_ = false;
  for (std::size_t iter = 1; iter < max_qp_iters_+1; ++iter){
    if (iter % rho_update_interval_ == 1 || iter == 1){
      backwardPass();
    }
    else{
      backwardPass_without_rho_update();
    }
    forwardPass();
    update_lagrangian_parameters(true);
    update_rho_sparse(iter);
    if(norm_primal_ < eps_abs_ + eps_rel_ * norm_primal_rel_ && norm_dual_ < eps_abs_ + eps_rel_ * norm_dual_rel_){
        qp_iters_ = iter;
        converged_ = true;
        break;
    }

    // if (iter % rho_update_interval_ == 0 && iter > 1){
    //   std::cout << "Iters " << iter << " res-primal " << norm_primal_ << " res-dual " << norm_dual_ << " optimal rho estimate " << rho_estimate_sparse_
    //           << " rho " << rho_sparse_ << std::endl;
    // }
  }

  if (!converged_){
    qp_iters_ = max_qp_iters_;
  }

  STOP_PROFILER("SolverCSQP::computeDirection");

}


void SolverCSQP::update_rho_sparse(int iter){
  double scale = (norm_primal_ * norm_dual_rel_)/ (norm_dual_ * norm_primal_rel_);
  scale = std::sqrt(scale);
  rho_estimate_sparse_ = scale * rho_sparse_;
  rho_estimate_sparse_ = std::min(std::max(rho_estimate_sparse_, rho_min_), rho_max_);

  if (iter % rho_update_interval_ == 0 && iter > 1){
    if(rho_estimate_sparse_ > rho_sparse_ * adaptive_rho_tolerance_ || 
            rho_estimate_sparse_ < rho_sparse_ / adaptive_rho_tolerance_){
      rho_sparse_ = rho_estimate_sparse_;
      const std::size_t T = problem_->get_T();
      // Running models
      for (std::size_t t = 0; t < T; ++t){
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem_->get_runningModels()[t];
        std::size_t nc = model->get_ng();
        auto lb = model->get_g_lb(); 
        auto ub = model->get_g_ub();
        double infty = std::numeric_limits<double>::infinity();

        for (std::size_t k = 0; k < nc; ++k){
          if (lb[k] == -infty && ub[k] == infty){
              rho_vec_[t][k] = rho_min_;
              inv_rho_vec_[t][k] = 1/rho_min_;
          }
          else if (abs(lb[k] - ub[k]) < 1e-6) {
              rho_vec_[t][k] = 1e3 * rho_sparse_;
              inv_rho_vec_[t][k] = 1/(1e3 * rho_sparse_);
          }
          else if (lb[k] < ub[k]){
              rho_vec_[t][k] = rho_sparse_;
              inv_rho_vec_[t][k] = 1/rho_sparse_;
          }
        }  
      }

      // Terminal model
      const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem_->get_terminalModel();
      std::size_t nc = model->get_ng();
      auto lb = model->get_g_lb(); 
      auto ub = model->get_g_ub();
      double infty = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < nc; ++k){
        if (lb[k] == -infty && ub[k] == infty){
            rho_vec_.back()[k] = rho_min_;
            inv_rho_vec_.back()[k] = 1/rho_min_;
        }
        else if (abs(lb[k] - ub[k]) < 1e-6) {
            rho_vec_.back()[k] = 1e3 * rho_sparse_;
            inv_rho_vec_.back()[k] = 1/(1e3 * rho_sparse_);
        }
        else if (lb[k] < ub[k]){
            rho_vec_.back()[k] = rho_sparse_;
            inv_rho_vec_.back()[k] = 1/rho_sparse_;
        }
      }  
    }
  }
}

void SolverCSQP::checkKKTConditions(){
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    if (t > 0){
      KKT_ = std::max(KKT_, (d->Lu + d->Fu.transpose() * lag_mul_[t+1] + d->Gu.transpose() * y_[t]).lpNorm<Eigen::Infinity>());
    }
    KKT_ = std::max(KKT_, (d->Lx + d->Fx.transpose() * lag_mul_[t+1] - lag_mul_[t] + d->Gx.transpose() * y_[t]).lpNorm<Eigen::Infinity>());
    fs_flat_.segment(t*ndx, ndx) = fs_[t];
  }
  fs_flat_.tail(ndx) = fs_.back();
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_ter = problem_->get_terminalData();
  KKT_ = std::max(KKT_, (d_ter->Lx - lag_mul_.back() + d_ter->Gx.transpose() * y_.back()).lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, constraint_norm_);
}


void SolverCSQP::forwardPass(){
    START_PROFILER("SolverCSQP::forwardPass");
    x_grad_norm_ = 0; u_grad_norm_ = 0;

    const std::size_t T = problem_->get_T();
    const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
      lag_mul_[t] = Vxx_[t] * dxtilde_[t] + Vx_[t];
      dutilde_[t].noalias() = -K_[t]*(dxtilde_[t]) - k_[t];
      dxtilde_[t+1].noalias() = (d->Fx - (d->Fu * K_[t]))*(dxtilde_[t]) - (d->Fu * (k_[t])) + fs_[t+1];
      
      x_grad_norm_ += dxtilde_[t].lpNorm<1>(); // assuming that there is no gap in the initial state
      u_grad_norm_ += dutilde_[t].lpNorm<1>();
    }
    
    lag_mul_.back() = Vxx_.back() * dxtilde_.back() + Vx_.back();

    x_grad_norm_ += dxtilde_.back().lpNorm<1>(); // assuming that there is no gap in the initial state
    x_grad_norm_ = x_grad_norm_/(T+1);
    u_grad_norm_ = u_grad_norm_/T; 
    STOP_PROFILER("SolverCSQP::forwardPass");

}

void SolverCSQP::backwardPass() {
  START_PROFILER("SolverCSQP::backwardPass");

  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_T = problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx + sigma_diag_x;
  Vx_.back() = d_T->Lx - sigma_ * dx_.back();

  if (problem_->get_terminalModel()->get_ng()){ // constraint model
    // TODO : make sure this is not used later
    Vxx_.back().noalias() += d_T->Gx.transpose() * rho_vec_.back().asDiagonal() * d_T->Gx;
    Vx_.back() += d_T->Gx.transpose() * (y_.back() - rho_vec_.back().cwiseProduct(z_.back()));
  }
  if (!std::isnan(xreg_)) {
    Vxx_.back().diagonal().array() += xreg_;
  }

  // if (!is_feasible_) {
  //   Vx_.back().noalias() += Vxx_.back() * fs_.back();
  // }



  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1] + Vxx_[t+1] * fs_[t+1];
    const std::size_t nu = m->get_nu();
    std::size_t nc = m->get_ng();
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    START_PROFILER("SolverCSQP::Qx");
    Qx_[t] = d->Lx - sigma_ * dx_[t];

    if (t > 0 && nc != 0){ //constraint model
      Qx_[t] += d->Gx.transpose() * (y_[t] -  rho_vec_[t].cwiseProduct(z_[t]));
    }
    
    Qx_[t].noalias() += d->Fx.transpose() * Vx_p;
    STOP_PROFILER("SolverCSQP::Qx");
    START_PROFILER("SolverCSQP::Qxx");
    Qxx_[t] = d->Lxx + sigma_diag_x;
    if (t > 0 && nc != 0){ //constraint model
      Qxx_[t].noalias() += d->Gx.transpose() * rho_vec_[t].asDiagonal() * d->Gx;

    }

    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    STOP_PROFILER("SolverCSQP::Qxx");
    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      START_PROFILER("SolverCSQP::Qu");
      Qu_[t] = d->Lu - sigma_ * du_[t];
      if (nc != 0){ //constraint model
        Qu_[t] += d->Gu.transpose() * (y_[t] - rho_vec_[t].cwiseProduct(z_[t]));
      }
      

      Qu_[t].noalias() += d->Fu.transpose() * Vx_p;

      STOP_PROFILER("SolverCSQP::Qu");
      START_PROFILER("SolverCSQP::Quu");
      Quu_[t] = d->Luu + sigma_diag_u[t];
      if (nc != 0){ //constraint model
        Quu_[t].noalias() += d->Gu.transpose() * rho_vec_[t].asDiagonal() * d->Gu;
      }

      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      STOP_PROFILER("SolverCSQP::Quu");
      START_PROFILER("SolverCSQP::Qxu");
      Qxu_[t] = d->Lxu; // TODO : check if this should also have added terms
      if (nc != 0){ //constraint model
        Qxu_[t].noalias() += d->Gx.transpose() * rho_vec_[t].asDiagonal() * d->Gu;
      }


      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
      STOP_PROFILER("SolverCSQP::Qxu");

      if (!std::isnan(ureg_)) {
        Quu_[t].diagonal().array() += ureg_;
      }
    }
    computeGains(t);
    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      START_PROFILER("SolverCSQP::Vxx");
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
      STOP_PROFILER("SolverCSQP::Vxx");
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;
    if (!std::isnan(xreg_)) {
      Vxx_[t].diagonal().array() += xreg_;
    }

    // Compute and store the Vx gradient at end of the interval (rollout state)
    // if (!is_feasible_) {
    //   Vx_[t].noalias() += Vxx_[t] * fs_[t];
    // }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
    if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
  STOP_PROFILER("SolverCSQP::backwardPass");
}

void SolverCSQP::backwardPass_without_constraints() {
  START_PROFILER("SolverCSQP::backwardPass_without_constraints");

  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_T = problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx;
  Vx_.back() = d_T->Lx ;

  if (!std::isnan(xreg_)) {
    Vxx_.back().diagonal().array() += xreg_;
  }

  // if (!is_feasible_) {
  //   Vx_.back().noalias() += Vxx_.back() * fs_.back();
  // }

  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1] + Vxx_[t+1] * fs_[t+1];;
    const std::size_t nu = m->get_nu();
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    START_PROFILER("SolverCSQP::Qx");
    Qx_[t] = d->Lx;

    Qx_[t].noalias() += d->Fx.transpose() * Vx_p;
    STOP_PROFILER("SolverCSQP::Qx");
    START_PROFILER("SolverCSQP::Qxx");
    Qxx_[t] = d->Lxx;
    
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    STOP_PROFILER("SolverCSQP::Qxx");
    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      START_PROFILER("SolverCSQP::Qu");
      Qu_[t] = d->Lu;
      Qu_[t].noalias() += d->Fu.transpose() * Vx_p;

      STOP_PROFILER("SolverCSQP::Qu");
      START_PROFILER("SolverCSQP::Quu");
      Quu_[t] = d->Luu;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      STOP_PROFILER("SolverCSQP::Quu");
      START_PROFILER("SolverCSQP::Qxu");
      Qxu_[t] = d->Lxu; // TODO : check if this should also have added terms
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
      STOP_PROFILER("SolverCSQP::Qxu");

      if (!std::isnan(ureg_)) {
        Quu_[t].diagonal().array() += ureg_;
      }
    }

    computeGains(t);

    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
      START_PROFILER("SolverCSQP::Vxx");
      Vxx_[t].noalias() -= Qxu_[t] * K_[t];
      STOP_PROFILER("SolverCSQP::Vxx");
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;

    if (!std::isnan(xreg_)) {
      Vxx_[t].diagonal().array() += xreg_;
    }

    // Compute and store the Vx gradient at end of the interval (rollout state)
    // if (!is_feasible_) {
    //   Vx_[t].noalias() += Vxx_[t] * fs_[t];
    // }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
    if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
  STOP_PROFILER("SolverCSQP::backwardPass_without_constraints");
}


void SolverCSQP::backwardPass_without_rho_update() {
  START_PROFILER("SolverCSQP::backwardPass_without_rho_update");

  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m_T = problem_->get_terminalModel();
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_T = problem_->get_terminalData();

  Vx_.back() = d_T->Lx - sigma_ * dx_.back();

  if (m_T->get_ng()){ // constraint model
    Vx_.back() += d_T->Gx.transpose() * (y_.back() - rho_vec_.back().cwiseProduct(z_.back()));
  }

  // if (!is_feasible_) {
  //   Vx_.back().noalias() += Vxx_.back() * fs_.back();
  // }

  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1] + Vxx_[t+1] * fs_[t+1];;
    const std::size_t nu = m->get_nu();
    std::size_t nc = m->get_ng();
    START_PROFILER("SolverCSQP::Qx");
    Qx_[t] = d->Lx - sigma_ * dx_[t];

    if (t > 0 && nc != 0){ //constraint model
      Qx_[t] += d->Gx.transpose() * (y_[t] - rho_vec_[t].cwiseProduct(z_[t]));
    }

    Qx_[t].noalias() += d->Fx.transpose() * Vx_p;

    STOP_PROFILER("SolverCSQP::Qxx");
    if (nu != 0) {
      START_PROFILER("SolverCSQP::Qu");
      Qu_[t] = d->Lu - sigma_ * du_[t];
      if (nc != 0){ //constraint model
        Qu_[t] += d->Gu.transpose() * (y_[t] - rho_vec_[t].cwiseProduct(z_[t]));
      }

      Qu_[t].noalias() += d->Fu.transpose() * Vx_p;

    }

    // computing gains efficiently
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);

    Vx_[t] = Qx_[t];
    if (nu != 0) {
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    }

    // Compute and store the Vx gradient at end of the interval (rollout state)
    // if (!is_feasible_) {
    //   Vx_[t].noalias() += Vxx_[t] * fs_[t];
    // }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
    if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
  STOP_PROFILER("SolverCSQP::backwardPass_without_rho_update");
}


void SolverCSQP::update_lagrangian_parameters(bool update_y){
    norm_primal_ = -1* std::numeric_limits<double>::infinity();
    norm_dual_ = -1* std::numeric_limits<double>::infinity();
    norm_primal_rel_ = -1* std::numeric_limits<double>::infinity();
    norm_dual_rel_ = -1* std::numeric_limits<double>::infinity();

    const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
    const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();

    const std::size_t T = problem_->get_T();
    for (std::size_t t = 0; t < T; ++t) {    

      const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
      const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

      std::size_t nc = m->get_ng();

      if (nc == 0){
        dx_[t] = dxtilde_[t];
        du_[t] = dutilde_[t];
        continue;
      }

      z_prev_[t] = z_[t];
      Cdx_Cdu[t] = d->Gx * dxtilde_[t] + d->Gu * dutilde_[t];
      z_relaxed_[t] = alpha_ * Cdx_Cdu[t] + (1 - alpha_) * z_[t];

      const auto ub = m->get_g_ub(); 
      const auto lb = m->get_g_lb();
      z_[t] = (z_relaxed_[t] + (y_[t].cwiseProduct(inv_rho_vec_[t])));
      z_[t] = z_[t].cwiseMax(lb - d->g).cwiseMin(ub - d->g);
     
      if (update_y){
        y_[t] = y_[t] + rho_vec_[t].cwiseProduct(z_relaxed_[t] - z_[t]);
      }
      dx_[t] = dxtilde_[t]; du_[t] = dutilde_[t];

      // operation repeated
      dual_vecx = d->Gx.transpose() * (rho_vec_[t].cwiseProduct(z_[t] - z_prev_[t]));
      dual_vecu = d->Gu.transpose() * (rho_vec_[t].cwiseProduct(z_[t] - z_prev_[t]));

      norm_dual_ = std::max(norm_dual_, std::max(dual_vecx.lpNorm<Eigen::Infinity>(), dual_vecu.lpNorm<Eigen::Infinity>()));
      norm_primal_ = std::max(norm_primal_, (Cdx_Cdu[t] - z_[t]).lpNorm<Eigen::Infinity>());
      norm_primal_rel_= std::max(norm_primal_rel_, Cdx_Cdu[t].lpNorm<Eigen::Infinity>());
      norm_primal_rel_= std::max(norm_primal_rel_, z_[t].lpNorm<Eigen::Infinity>());
      norm_dual_rel_ = std::max(norm_dual_rel_, (d->Gx.transpose() * y_[t]).lpNorm<Eigen::Infinity>());
      norm_dual_rel_ = std::max(norm_dual_rel_, (d->Gu.transpose() * y_[t]).lpNorm<Eigen::Infinity>());
    }

  dx_.back() = dxtilde_.back();
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m_T = problem_->get_terminalModel();
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_T = problem_->get_terminalData();
  std::size_t nc = m_T->get_ng();

  if (nc != 0){
    z_prev_.back() = z_.back();
    Cdx_Cdu.back() = d_T->Gx * dxtilde_.back() ;

    z_relaxed_.back() = alpha_ * Cdx_Cdu.back() + (1 - alpha_) * z_.back();
    auto ub = m_T->get_g_ub(); 
    auto lb = m_T->get_g_lb(); 
    z_.back() = (z_relaxed_.back() + (y_.back().cwiseProduct(inv_rho_vec_.back())));
    z_.back() = z_.back().cwiseMax(lb - d_T->g).cwiseMin(ub - d_T->g);

    if (update_y){
      y_.back() = y_.back() + rho_vec_.back().cwiseProduct(z_relaxed_.back() - z_.back());
    }
    dx_.back() = dxtilde_.back();

    dual_vecx = d_T->Gx.transpose() * rho_vec_.back().cwiseProduct(z_.back() - z_prev_.back());

    norm_dual_ = std::max(norm_dual_, dual_vecx.lpNorm<Eigen::Infinity>());
    norm_primal_ = std::max(norm_primal_, (Cdx_Cdu.back() - z_.back()).lpNorm<Eigen::Infinity>());
    norm_primal_rel_= std::max(norm_primal_rel_, Cdx_Cdu.back().lpNorm<Eigen::Infinity>());
    norm_primal_rel_= std::max(norm_primal_rel_, z_.back().lpNorm<Eigen::Infinity>());
    norm_dual_rel_ = std::max(norm_dual_rel_, (d_T->Gx.transpose() * y_.back()).lpNorm<Eigen::Infinity>());
  }

}

double SolverCSQP::tryStep(const double steplength) {
    if (steplength > 1. || steplength < 0.) {
        throw_pretty("Invalid argument: "
                    << "invalid step length, value is between 0. to 1.");
    }

    START_PROFILER("SolverCSQP::tryStep");
    cost_try_ = 0.;
    merit_try_ = 0;
    gap_norm_try_ = 0;
    constraint_norm_try_ = 0;
    
    const std::size_t T = problem_->get_T();
    const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
    const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();

    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
      m->get_state()->integrate(xs_[t], steplength * dx_[t], xs_try_[t]); 
      const std::size_t nu = m->get_nu();

      if (nu != 0) {
        us_try_[t].noalias() = us_[t] + steplength * du_[t];
        }        
      } 

    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m_ter = problem_->get_terminalModel();
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_ter = problem_->get_terminalData();

    m_ter->get_state()->integrate(xs_.back(), steplength * dx_.back(), xs_try_.back()); 

    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
      const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    
      m->calc(d, xs_try_[t], us_try_[t]);        
      cost_try_ += d->cost;      
      m->get_state()->diff(xs_try_[t+1], d->xnext, fs_try_[t+1]);
      gap_norm_try_ += fs_try_[t+1].lpNorm<1>(); 

      std::size_t nc = m->get_ng();
      auto lb = m->get_g_lb(); 
      auto ub = m->get_g_ub();
      constraint_norm_try_ += (lb - d->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
      constraint_norm_try_ += (d->g - ub).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

      if (raiseIfNaN(cost_try_)) {
        STOP_PROFILER("SolverCSQP::tryStep");
        throw_pretty("step_error");
      }   
    }

    // Terminal state update
    m_ter->calc(d_ter, xs_try_.back());
    cost_try_ += d_ter->cost;

    std::size_t nc = m_ter->get_ng();
    auto lb = m_ter->get_g_lb(); 
    auto ub = m_ter->get_g_ub();

    constraint_norm_try_ += (lb - d_ter->g).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();
    constraint_norm_try_ += (d_ter->g - ub).cwiseMax(Eigen::VectorXd::Zero(nc)).lpNorm<1>();

    merit_try_ = cost_try_ + mu_*gap_norm_try_ + mu2_*constraint_norm_try_;

    if (raiseIfNaN(cost_try_)) {
        STOP_PROFILER("SolverCSQP::tryStep");
        throw_pretty("step_error");
    }

    STOP_PROFILER("SolverCSQP::tryStep");

    return merit_try_;
}

void SolverCSQP::printCallbacks(){
  if (this->get_iter() % 10 == 0) {
    std::cout << "iter     merit        cost         grad       step     ||gaps||       KKT       Constraint Norms    QP Iters";
    std::cout << std::endl;
  }
  std::cout << std::setw(4) << this->get_iter() << "  ";
  std::cout << std::scientific << std::setprecision(5) << this->get_merit() << "  ";
  std::cout << std::scientific << std::setprecision(5) << this->get_cost() << "  ";
  std::cout << this->get_xgrad_norm() + this->get_ugrad_norm() << "  ";
  std::cout << std::fixed << std::setprecision(4) << this->get_steplength() << "  ";
  std::cout << std::scientific << std::setprecision(5) << this->get_gap_norm() << "  ";
  std::cout << std::scientific << std::setprecision(5) << KKT_ << "    ";
  std::cout << std::scientific << std::setprecision(5) << constraint_norm_ << "         ";
  std::cout << std::scientific << std::setprecision(5) << qp_iters_;

  std::cout << std::endl;
  std::cout << std::flush;
}

void SolverCSQP::setCallbacks(bool inCallbacks){
  with_callbacks_ = inCallbacks;
}

bool SolverCSQP::getCallbacks(){
  return with_callbacks_;
}
// double SolverCSQP::get_th_acceptnegstep() const { return th_acceptnegstep_; }

// void SolverCSQP::set_th_acceptnegstep(const double th_acceptnegstep) {
//   if (0. > th_acceptnegstep) {
//     throw_pretty("Invalid argument: "
//                  << "th_acceptnegstep value has to be positive.");
//   }
//   th_acceptnegstep_ = th_acceptnegstep;
// }

}  // namespace mim_solvers
