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

#include <iostream>
#include <iomanip>

#include <crocoddyl/core/utils/exception.hpp>
#include "mim_solvers/csqp.hpp"

using namespace crocoddyl;

namespace mim_solvers {


SolverCSQP::SolverCSQP(boost::shared_ptr<crocoddyl::ShootingProblem> problem)
    : SolverDDP(problem){
      
      const std::size_t T = this->problem_->get_T();
      const std::size_t ndx = problem_->get_ndx();
      constraint_list_.resize(filter_size_);
      gap_list_.resize(filter_size_);
      cost_list_.resize(filter_size_);

      fs_flat_.resize(ndx*(T + 1));
      fs_flat_.setZero();
      
      xs_try_.resize(T+1); 
      us_try_.resize(T);
      dx_.resize(T+1); 
      du_.resize(T);
      dxtilde_.resize(T+1);
      dutilde_.resize(T);
      lag_mul_.resize(T+1);
      fs_try_.resize(T + 1);
      
      z_.resize(T+1); 
      z_relaxed_.resize(T+1); 
      z_prev_.resize(T+1);
      y_.resize(T+1); 
      rho_vec_.resize(T+1); 
      inv_rho_vec_.resize(T+1);
      rho_sparse_ = rho_sparse_base_;
      std::size_t n_eq_crocoddyl = 0;
      
      tmp_Vx_.resize(ndx); 
      tmp_Vx_.setZero();
      tmp_vec_x_.resize(ndx);
      tmp_vec_x_.setZero();

      tmp_Cdx_Cdu_.resize(T+1);
      tmp_dual_cwise_.resize(T+1);
      tmp_rhoGx_mat_.resize(T+1);
      tmp_rhoGu_mat_.resize(T);
      tmp_vec_u_.resize(T);
      Vxx_fs_.resize(T);

      const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
      for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
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

      std::size_t nc = problem_->get_terminalModel()->get_ng(); 

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
      if(n_eq_crocoddyl != 0){
        throw_pretty("Error: nh must be zero !!! Crocoddyl's equality constraints API is not supported by mim_solvers.\n"
                     "  >> Equality constraints of the form H(x,u) = h must be implemented as g <= G(x,u) <= g by specifying \n"
                     "     lower and upper bounds in the constructor of the constraint model residual or by setting g_ub and g_lb.")
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

  if (reset_rho_){
    reset_rho_vec();
  }
  
  const std::size_t T = this->problem_->get_T();

  for (std::size_t t = 0; t < T; ++t) {

    z_[t].setZero();
    z_prev_[t].setZero();
    z_relaxed_[t].setZero();

    if(reset_y_){
      y_[t].setZero();
    } 
  }
  z_.back().setZero();
  z_prev_.back().setZero();
  z_relaxed_.back().setZero();

  if(reset_y_){
    y_.back().setZero();
  }
}


SolverCSQP::~SolverCSQP() {}

bool SolverCSQP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                       const std::size_t maxiter, const bool is_feasible, const double reginit) {
  
  (void)is_feasible;

  START_PROFILER("SolverCSQP::solve");
  if (problem_->is_updated()) {
    resizeData();
  }
  setCandidate(init_xs, init_us, false);
  xs_[0] = problem_->get_x0();      // Otherwise xs[0] is overwritten by init_xs inside setCandidate()
  xs_try_[0] = problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible

  // Optionally remove Crocoddyl's regularization
  if(remove_reg_){
    preg_ = 0.;
    dreg_ = 0;
  } 
  else {
    if (std::isnan(reginit)) {
      preg_ = reg_min_;
      dreg_ = reg_min_;
    } else {
      preg_ = reginit;
      dreg_ = reginit;
    }
  }

  // Main SQP loop
  for (iter_ = 0; iter_ < maxiter; ++iter_) {

    // Compute gradients
    calc(true);

    // reset rho only at the beginning of each solve if reset_rho_ is false 
    // (after calc to get correct lb and ub)
    if (iter_ == 0 && !reset_rho_){
      reset_rho_vec();
    }

    // Solve QP
    if(remove_reg_){
      computeDirection(true);
    } else {
      while (true) {
        try {
          computeDirection(true);
        } 
        catch (std::exception& e) {
          increaseRegularization();
          if (preg_ == reg_max_) {
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
    if (KKT_  <= termination_tol_) {
      if(with_callbacks_){
        printCallbacks();
      }
      STOP_PROFILER("SolverCSQP::solve");
      return true;
    }
  

    // Line search
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
      if(use_filter_line_search_){
        is_worse_than_memory_ = false;
        std::size_t count = 0.; 
        while( count < filter_size_ && is_worse_than_memory_ == false and count <= iter_){
          is_worse_than_memory_ = cost_list_[filter_size_-1-count] <= cost_try_ && gap_list_[filter_size_-1-count] <= gap_norm_try_ && constraint_list_[filter_size_-1-count] <= constraint_norm_try_;
          count++;
        }
        if( is_worse_than_memory_ == false ) {
          setCandidate(xs_try_, us_try_, false);
          break;
        } 
      }
      // Line-search criteria using merit function
      else{
        if (merit_ > merit_try_) {
          setCandidate(xs_try_, us_try_, false);
          break;
        }
      }
    }

    // Regularization logic
    if(remove_reg_ == false){
      if (steplength_ > th_stepdec_) {
        decreaseRegularization();
      }
      if (steplength_ <= th_stepinc_) {
        increaseRegularization();
        if (preg_ == reg_max_) {
          STOP_PROFILER("SolverCSQP::solve");
          return false;
        }
      }
    }

    // Print
    if(with_callbacks_){
      printCallbacks();
    }
  }

  // If reached max iter, still compute KKT residual
  if (extra_iteration_for_last_kkt_){
    // Compute gradients
    calc(true);

    // Solve QP
    if(remove_reg_){
      computeDirection(true);
    } else {
      while (true) {
        try {
          computeDirection(true);
        } 
        catch (std::exception& e) {
          increaseRegularization();
          if (preg_ == reg_max_) {
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
    if(with_callbacks_){
      printCallbacks(true);
    }

    if (KKT_  <= termination_tol_) {
      STOP_PROFILER("SolverCSQP::solve");
      return true;
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

  (void)recalcDiff;

  reset_params();

  if (equality_qp_initial_guess_){
    backwardPass_without_constraints();
    forwardPass_without_constraints();
  }

  if(with_qp_callbacks_){
    printQPCallbacks(0);
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
    update_lagrangian_parameters();
    update_rho_vec(iter);
    // Because (eps_rel=0) x inf = NaN
    if(eps_rel_ == 0){
      norm_primal_tolerance_ = eps_abs_;
      norm_dual_tolerance_   = eps_abs_;
    } 
    else{
      norm_primal_tolerance_ = eps_abs_ + eps_rel_ * norm_primal_rel_;
      norm_dual_tolerance_   = eps_abs_ + eps_rel_ * norm_dual_rel_;
    }
    if(norm_primal_ <= norm_primal_tolerance_ && norm_dual_ <= norm_dual_tolerance_){
        qp_iters_ = iter;
        converged_ = true;
        break;
    }
    if(with_qp_callbacks_){
      printQPCallbacks(iter);
    }
  }

  if (!converged_){
    qp_iters_ = max_qp_iters_;
  }

  STOP_PROFILER("SolverCSQP::computeDirection");

}

void SolverCSQP::update_rho_vec(int iter){
  double scale = (norm_primal_ * norm_dual_rel_)/ (norm_dual_ * norm_primal_rel_);
  scale = std::sqrt(scale);
  rho_estimate_sparse_ = scale * rho_sparse_;
  rho_estimate_sparse_ = std::min(std::max(rho_estimate_sparse_, rho_min_), rho_max_);


  if (iter % rho_update_interval_ == 0 && iter > 1){
    if(rho_estimate_sparse_ > rho_sparse_ * adaptive_rho_tolerance_ || 
            rho_estimate_sparse_ < rho_sparse_ / adaptive_rho_tolerance_){
      rho_sparse_ = rho_estimate_sparse_;
      apply_rho_update(rho_sparse_);
    }  
  }
}


void SolverCSQP::reset_rho_vec(){
  rho_sparse_ = rho_sparse_base_;
  apply_rho_update(rho_sparse_);
}


void SolverCSQP::apply_rho_update(double rho_sparse_){
  const std::size_t T = this->problem_->get_T();
  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
  double infty = std::numeric_limits<double>::infinity();

  for (std::size_t t = 0; t < T; ++t) {

    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    std::size_t nc = m->get_ng();
    const auto ub = m->get_g_ub(); 
    const auto lb = m->get_g_lb();

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

  std::size_t nc = problem_->get_terminalModel()->get_ng();
  auto lb = problem_->get_terminalModel()->get_g_lb(); 
  auto ub = problem_->get_terminalModel()->get_g_ub();
  
  for (std::size_t k = 0; k < nc; ++k){
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

void SolverCSQP::checkKKTConditions(){
  KKT_ = 0.;
  const std::size_t T = problem_->get_T();

  for (std::size_t t = 0; t < T; ++t) {
    lag_mul_[t].noalias() = Vx_[t];
    lag_mul_[t].noalias() += Vxx_[t] * dxtilde_[t];
  }
  lag_mul_.back().noalias() = Vx_.back();
  lag_mul_.back().noalias() += Vxx_.back() * dxtilde_.back() ;
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
    tmp_vec_x_ = d->Lx;
    tmp_vec_x_.noalias() += d->Fx.transpose() * lag_mul_[t+1];
    tmp_vec_x_ -= lag_mul_[t];
    if (t > 0){
      tmp_vec_x_.noalias() += d->Gx.transpose() * y_[t];
    }
    KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
    tmp_vec_u_[t] = d->Lu;
    tmp_vec_u_[t].noalias() += d->Fu.transpose() * lag_mul_[t+1];
    tmp_vec_u_[t].noalias() += d->Gu.transpose() * y_[t];
    KKT_ = std::max(KKT_, tmp_vec_u_[t].lpNorm<Eigen::Infinity>());
    fs_flat_.segment(t*ndx, ndx) = fs_[t];
  }
  fs_flat_.tail(ndx) = fs_.back();
  const boost::shared_ptr<ActionDataAbstract>& d_ter = problem_->get_terminalData();
  tmp_vec_x_ = d_ter->Lx;
  tmp_vec_x_ -= lag_mul_.back();
  tmp_vec_x_ += d_ter->Gx.transpose() * y_.back();
  KKT_ = std::max(KKT_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, fs_flat_.lpNorm<Eigen::Infinity>());
  KKT_ = std::max(KKT_, constraint_norm_);
}


void SolverCSQP::forwardPass(const double stepLength){

    (void)stepLength;

    START_PROFILER("SolverCSQP::forwardPass");
    x_grad_norm_ = 0; 
    u_grad_norm_ = 0;

    const std::size_t T = problem_->get_T();
    const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

      dutilde_[t].noalias() = -K_[t] * dxtilde_[t];
      dutilde_[t].noalias() -= k_[t];
      dxtilde_[t+1].noalias() = d->Fx * dxtilde_[t];
      dxtilde_[t+1].noalias() += d->Fu * dutilde_[t];
      dxtilde_[t+1].noalias() += fs_[t+1];

      x_grad_norm_ += dxtilde_[t].lpNorm<1>(); 
      u_grad_norm_ += dutilde_[t].lpNorm<1>();
    }

    x_grad_norm_ += dxtilde_.back().lpNorm<1>(); 
    x_grad_norm_ = x_grad_norm_/(T+1);
    u_grad_norm_ = u_grad_norm_/T; 
    STOP_PROFILER("SolverCSQP::forwardPass");

}


void SolverCSQP::forwardPass_without_constraints(){
    START_PROFILER("SolverCSQP::forwardPass_without_constraints");

    const std::size_t T = problem_->get_T();
    const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

      du_[t].noalias() = -K_[t] * dx_[t];
      du_[t].noalias() -= k_[t];
      dx_[t+1].noalias() = d->Fx * dx_[t];
      dx_[t+1].noalias() += d->Fu * du_[t];
      dx_[t+1].noalias() += fs_[t+1];
    }

    STOP_PROFILER("SolverCSQP::forwardPass_without_constraints");

}



void SolverCSQP::backwardPass() {
  START_PROFILER("SolverCSQP::backwardPass");

  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_T = problem_->get_terminalData();

  Vxx_.back() = d_T->Lxx; 
  Vxx_.back().diagonal().array() += sigma_;
  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= sigma_ * dx_.back();

  if (problem_->get_terminalModel()->get_ng()){ 
    tmp_rhoGx_mat_.back().noalias() = rho_vec_.back().asDiagonal() * d_T->Gx;
    Vxx_.back().noalias() += d_T->Gx.transpose() * tmp_rhoGx_mat_.back();
    tmp_dual_cwise_.back() = y_.back() - rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }
  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }


  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];

    Vxx_fs_[t].noalias() = Vxx_[t+1] * fs_[t+1];
    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];

    const std::size_t nu = m->get_nu();
    std::size_t nc = m->get_ng();
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    START_PROFILER("SolverCSQP::Qx");
    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= sigma_ * dx_[t];
    if (t > 0 && nc != 0){ 
      tmp_dual_cwise_[t] = y_[t] - rho_vec_[t].cwiseProduct(z_[t]);
      Qx_[t] += d->Gx.transpose() * tmp_dual_cwise_[t];
    }
    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    STOP_PROFILER("SolverCSQP::Qx");

    START_PROFILER("SolverCSQP::Qxx");
    Qxx_[t] = d->Lxx; 
    Qxx_[t].diagonal().array() += sigma_;
    if (t > 0 && nc != 0){ 
      tmp_rhoGx_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gx;
      Qxx_[t].noalias() += d->Gx.transpose() * tmp_rhoGx_mat_[t];
    }
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    STOP_PROFILER("SolverCSQP::Qxx");

    if (nu != 0) {
      START_PROFILER("SolverCSQP::Qu");
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      Qu_[t] = d->Lu - sigma_ * du_[t];
      if (nc != 0){ 
        tmp_dual_cwise_[t] = y_[t] - rho_vec_[t].cwiseProduct(z_[t]);
        Qu_[t] += d->Gu.transpose() * tmp_dual_cwise_[t];
      }
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;
      STOP_PROFILER("SolverCSQP::Qu");

      START_PROFILER("SolverCSQP::Quu");
      Quu_[t] = d->Luu; 
      Quu_[t].diagonal().array() += sigma_;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      if (nc != 0){ 
        tmp_rhoGu_mat_[t].noalias() = rho_vec_[t].asDiagonal() * d->Gu;
        Quu_[t].noalias() += d->Gu.transpose() * tmp_rhoGu_mat_[t];
      }
      STOP_PROFILER("SolverCSQP::Quu");

      START_PROFILER("SolverCSQP::Qxu");
      Qxu_[t] = d->Lxu;
      if (t > 0 && nc != 0){ 
        Qxu_[t].noalias() += d->Gx.transpose() * tmp_rhoGu_mat_[t];
      }
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
      STOP_PROFILER("SolverCSQP::Qxu");

      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
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
    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
    }

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

  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }

  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    tmp_Vx_.noalias() = Vxx_[t+1] * fs_[t+1];
    tmp_Vx_.noalias() += Vx_[t + 1];
    const std::size_t nu = m->get_nu();
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    START_PROFILER("SolverCSQP::Qx");
    Qx_[t] = d->Lx;

    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;
    STOP_PROFILER("SolverCSQP::Qx");
    START_PROFILER("SolverCSQP::Qxx");
    Qxx_[t] = d->Lxx;
    
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    STOP_PROFILER("SolverCSQP::Qxx");
    if (nu != 0) {
      FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
      START_PROFILER("SolverCSQP::Qu");
      Qu_[t] = d->Lu;
      Qu_[t].noalias() += d->Fu.transpose() * tmp_Vx_;

      STOP_PROFILER("SolverCSQP::Qu");
      START_PROFILER("SolverCSQP::Quu");
      Quu_[t] = d->Luu;
      Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
      STOP_PROFILER("SolverCSQP::Quu");
      START_PROFILER("SolverCSQP::Qxu");
      Qxu_[t] = d->Lxu; 
      Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
      STOP_PROFILER("SolverCSQP::Qxu");

      if (!std::isnan(dreg_)) {
        Quu_[t].diagonal().array() += dreg_;
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

    if (!std::isnan(preg_)) {
      Vxx_[t].diagonal().array() += preg_;
    }

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

  Vx_.back() = d_T->Lx;
  Vx_.back().noalias() -= sigma_ * dx_.back();

  if (m_T->get_ng()){ // constraint model
    tmp_dual_cwise_.back() = y_.back();
    tmp_dual_cwise_.back().noalias() -= rho_vec_.back().cwiseProduct(z_.back());
    Vx_.back().noalias() += d_T->Gx.transpose() * tmp_dual_cwise_.back();
  }

  const std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<crocoddyl::ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d = datas[t];

    tmp_Vx_ = Vxx_fs_[t] + Vx_[t + 1];
    const std::size_t nu = m->get_nu();
    std::size_t nc = m->get_ng();
    START_PROFILER("SolverCSQP::Qx");
    Qx_[t] = d->Lx;
    Qx_[t].noalias() -= sigma_ * dx_[t];

    if (t > 0 && nc != 0){ 
      tmp_dual_cwise_[t] = y_[t]; 
      tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
      Qx_[t].noalias() += d->Gx.transpose() * tmp_dual_cwise_[t];
    }

    Qx_[t].noalias() += d->Fx.transpose() * tmp_Vx_;

    STOP_PROFILER("SolverCSQP::Qxx");
    if (nu != 0) {
      START_PROFILER("SolverCSQP::Qu");
      Qu_[t] = d->Lu;
      Qu_[t].noalias() -= sigma_ * du_[t];
      if (nc != 0){ 
        tmp_dual_cwise_[t] = y_[t]; 
        tmp_dual_cwise_[t].noalias() -= rho_vec_[t].cwiseProduct(z_[t]);
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

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
    if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
  STOP_PROFILER("SolverCSQP::backwardPass_without_rho_update");
}


void SolverCSQP::update_lagrangian_parameters(){
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
      tmp_Cdx_Cdu_[t].noalias() = d->Gx * dxtilde_[t];
      tmp_Cdx_Cdu_[t].noalias() += d->Gu * dutilde_[t];
      z_relaxed_[t].noalias() = alpha_ * tmp_Cdx_Cdu_[t];
      z_relaxed_[t].noalias() += (1 - alpha_) * z_[t];


      const auto ub = m->get_g_ub(); 
      const auto lb = m->get_g_lb();


      tmp_dual_cwise_[t] = y_[t].cwiseProduct(inv_rho_vec_[t]);

      z_[t] = (z_relaxed_[t] + tmp_dual_cwise_[t]);
      z_[t] = z_[t].cwiseMax(lb - d->g).cwiseMin(ub - d->g);
     
      
      y_[t] += rho_vec_[t].cwiseProduct(z_relaxed_[t] - z_[t]);
      
      dx_[t] = dxtilde_[t];
      du_[t] = dutilde_[t];

      if (update_rho_with_heuristic_){
        tmp_dual_cwise_[t] = rho_vec_[t].cwiseProduct(z_[t] - z_prev_[t]);
        norm_dual_ = std::max(norm_dual_, tmp_dual_cwise_[t].lpNorm<Eigen::Infinity>());
        norm_primal_ = std::max(norm_primal_, (tmp_Cdx_Cdu_[t] - z_[t]).lpNorm<Eigen::Infinity>());

        norm_primal_rel_= std::max(norm_primal_rel_, tmp_Cdx_Cdu_[t].lpNorm<Eigen::Infinity>());
        norm_primal_rel_= std::max(norm_primal_rel_, z_[t].lpNorm<Eigen::Infinity>());
        norm_dual_rel_ = std::max(norm_dual_rel_, y_[t].lpNorm<Eigen::Infinity>());
      } 
      else {
        tmp_dual_cwise_[t] = rho_vec_[t].cwiseProduct(z_[t] - z_prev_[t]);
        tmp_vec_x_.noalias() = d->Gx.transpose() * tmp_dual_cwise_[t];
        tmp_vec_u_[t].noalias() = d->Gu.transpose() * tmp_dual_cwise_[t];
        norm_dual_ = std::max(norm_dual_, std::max(tmp_vec_x_.lpNorm<Eigen::Infinity>(), tmp_vec_u_[t].lpNorm<Eigen::Infinity>()));
        norm_primal_ = std::max(norm_primal_, (tmp_Cdx_Cdu_[t] - z_[t]).lpNorm<Eigen::Infinity>());
        
        norm_primal_rel_= std::max(norm_primal_rel_, tmp_Cdx_Cdu_[t].lpNorm<Eigen::Infinity>());
        norm_primal_rel_= std::max(norm_primal_rel_, z_[t].lpNorm<Eigen::Infinity>());
        tmp_vec_x_.noalias() = d->Gx.transpose() * y_[t];
        tmp_vec_u_[t].noalias() = d->Gu.transpose() * y_[t];
        norm_dual_rel_ = std::max(norm_dual_rel_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
        norm_dual_rel_ = std::max(norm_dual_rel_, tmp_vec_u_[t].lpNorm<Eigen::Infinity>());
      }
    }

  dx_.back() = dxtilde_.back();
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& m_T = problem_->get_terminalModel();
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& d_T = problem_->get_terminalData();
  std::size_t nc = m_T->get_ng();

  if (nc != 0){
    z_prev_.back() = z_.back();
    tmp_Cdx_Cdu_.back().noalias() = d_T->Gx * dxtilde_.back() ;
    z_relaxed_.back().noalias() = alpha_ * tmp_Cdx_Cdu_.back();
    z_relaxed_.back().noalias() += (1 - alpha_) * z_.back();

    auto ub = m_T->get_g_ub(); 
    auto lb = m_T->get_g_lb(); 

    tmp_dual_cwise_.back() = y_.back().cwiseProduct(inv_rho_vec_.back());
    z_.back() = (z_relaxed_.back() + tmp_dual_cwise_.back());
    z_.back() = z_.back().cwiseMax(lb - d_T->g).cwiseMin(ub - d_T->g);
    y_.back() += rho_vec_.back().cwiseProduct(z_relaxed_.back() - z_.back());
    

    if (update_rho_with_heuristic_){
      tmp_dual_cwise_.back() = rho_vec_.back().cwiseProduct(z_.back() - z_prev_.back());
      norm_dual_ = std::max(norm_dual_, tmp_dual_cwise_.back().lpNorm<Eigen::Infinity>());
      norm_primal_ = std::max(norm_primal_, (tmp_Cdx_Cdu_.back() - z_.back()).lpNorm<Eigen::Infinity>());

      norm_primal_rel_= std::max(norm_primal_rel_, tmp_Cdx_Cdu_.back().lpNorm<Eigen::Infinity>());
      norm_primal_rel_= std::max(norm_primal_rel_, z_.back().lpNorm<Eigen::Infinity>());
      norm_dual_rel_ = std::max(norm_dual_rel_, y_.back().lpNorm<Eigen::Infinity>());
    }
    else {
      tmp_dual_cwise_.back() = rho_vec_.back().cwiseProduct(z_.back() - z_prev_.back());
      tmp_vec_x_.noalias() = d_T->Gx.transpose() * tmp_dual_cwise_.back();
      norm_dual_ = std::max(norm_dual_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
      norm_primal_ = std::max(norm_primal_, (tmp_Cdx_Cdu_.back() - z_.back()).lpNorm<Eigen::Infinity>());

      norm_primal_rel_= std::max(norm_primal_rel_, tmp_Cdx_Cdu_.back().lpNorm<Eigen::Infinity>());
      norm_primal_rel_= std::max(norm_primal_rel_, z_.back().lpNorm<Eigen::Infinity>());
      tmp_vec_x_.noalias() = d_T->Gx.transpose() * y_.back();
      norm_dual_rel_ = std::max(norm_dual_rel_, tmp_vec_x_.lpNorm<Eigen::Infinity>());
    }
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
        us_try_[t] = us_[t];
        us_try_[t].noalias() += steplength * du_[t];
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

void SolverCSQP::printCallbacks(bool isLastIteration){
  if (this->get_iter() % 10 == 0) {
    std::cout << std::scientific << std::setprecision(4) <<  "iter"               << "  "; // Iteration number
    std::cout << std::scientific << std::setprecision(4) <<  "   merit"           << "  "; // Merit function value
    std::cout << std::scientific << std::setprecision(4) <<  "      cost"         << "  "; // Cost function value
    std::cout << std::scientific << std::setprecision(4) <<  "     ||gaps||"      << "  "; // Gaps norm 
    std::cout << std::scientific << std::setprecision(4) <<  " ||Constraint||"    << "";   // Constraint norm 
    std::cout << std::scientific << std::setprecision(4) <<  "  ||(dx,du)||"      << " ";  // Step norm
    std::cout << std::fixed      << std::setprecision(4) <<  "    step"           << "  "; // Step size
    std::cout << std::scientific << std::setprecision(4) <<  " KKT criteria"      << "  "; // KKT residual norm
    std::cout << std::fixed      << std::setprecision(4) <<  "QP iters"           << " ";  // Number of QP iterations
    std::cout << std::endl;
  }
  if(KKT_ < termination_tol_ || isLastIteration){
    std::cout << std::setw(4) << "END" << "  ";
    std::cout << std::scientific << std::setprecision(4) << this->get_merit()     << "   ";    
    std::cout << std::scientific << std::setprecision(4) << this->get_cost()      << "   ";
    std::cout << std::scientific << std::setprecision(4) << this->get_gap_norm()  << "    ";
    std::cout << std::scientific << std::setprecision(4) << constraint_norm_      << "    ";
    std::cout << std::fixed <<      std::setprecision(5) << "   ---- "            << "    ";
    std::cout << std::scientific << std::setprecision(4) << "    ---- "           << "   ";
    std::cout << std::scientific << std::setprecision(4) << KKT_                  << "    ";
    std::cout << std::fixed <<      std::setprecision(4) << "-----";
  } else {
    std::cout << std::setw(4) << this->get_iter()+1 << "  ";
    std::cout << std::scientific << std::setprecision(4) << this->get_merit()                                     << "   ";
    std::cout << std::scientific << std::setprecision(4) << this->get_cost()                                      << "   ";
    std::cout << std::scientific << std::setprecision(4) << this->get_gap_norm()                                  << "    ";
    std::cout << std::scientific << std::setprecision(4) << constraint_norm_                                      << "    ";
    std::cout << std::scientific << std::setprecision(5) << (this->get_xgrad_norm() + this->get_ugrad_norm()) / 2 << "    ";
    std::cout << std::fixed      << std::setprecision(4) << this->get_steplength()                                << "   ";
    if(iter_ == 0){
      std::cout << std::scientific << std::setprecision(4) << "   ----   "                                        << "     ";
    } else {
      std::cout << std::scientific << std::setprecision(4) << KKT_                                                << "     ";
    }
    std::cout << std::fixed      << std::setprecision(4) << qp_iters_;
  }
  std::cout << std::endl;
  std::cout << std::flush;
}

void SolverCSQP::printQPCallbacks(int iter){
  std::cout << "Iters " << iter;
  std::cout << " norm_primal = "     << std::scientific << std::setprecision(4) << norm_primal_;
  std::cout << " norm_primal_tol = " << std::scientific << std::setprecision(4) << norm_primal_tolerance_;
  std::cout << " norm_dual =  "      << std::scientific << std::setprecision(4) << norm_dual_;
  std::cout << " norm_dual_tol = "   << std::scientific << std::setprecision(4) << norm_dual_tolerance_;
  std::cout << std::endl;
  std::cout << std::flush;
}

void SolverCSQP::setCallbacks(bool inCallbacks){
  with_callbacks_ = inCallbacks;
}

bool SolverCSQP::getCallbacks(){
  return with_callbacks_;
}

void SolverCSQP::setQPCallbacks(bool inQPCallbacks){
  with_qp_callbacks_ = inQPCallbacks;
}

bool SolverCSQP::getQPCallbacks(){
  return with_qp_callbacks_;
}


}  // namespace mim_solvers
