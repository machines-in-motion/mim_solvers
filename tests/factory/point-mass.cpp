///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <crocoddyl/core/states/euclidean.hpp>

#include "point-mass.hpp"


namespace mim_solvers {
namespace unittest {

DAMPointMass1D::DAMPointMass1D(const std::size_t ng,
                               const bool x_eq,
                               const bool x_ineq,
                               const bool u_eq,
                               const bool u_ineq,
                               const bool isInitial,
                               const bool isTerminal)
    : DAMBase(boost::make_shared<crocoddyl::StateVector>(2), 1, 1, ng) {

  std::cout << "ng         = " << ng << std::endl;
  std::cout << "x_eq       = " << x_eq << std::endl;
  std::cout << "x_ineq     = " << x_ineq << std::endl;
  std::cout << "u_eq       = " << u_eq << std::endl;
  std::cout << "u_ineq     = " << u_ineq << std::endl;
  std::cout << "isInitial  = " << isInitial << std::endl;
  std::cout << "isTerminal = " << isTerminal << std::endl;

  // Check constraint setup 
  if((u_eq && isTerminal)||(u_ineq && isTerminal)){
    std::cout << "Error : terminal model cannot have a control constraint !" << std::endl;
  }
  if((x_eq && isInitial)||(x_ineq && isInitial)){
    std::cout << "Error : initial model cannot have a state constraint !" << std::endl;
  }
  if((u_eq && u_ineq)||(x_eq && x_ineq)){
    std::cout << "Error: test case unsupported (too many constraints) !" << std::endl;
  }
  isTerminal_ = isTerminal;
  isInitial_ = isInitial;
  // Cost and dynamics parameters
  gravity_ = Eigen::Vector2d(0,-9.81);
  x_weights_terminal_ = Eigen::Vector2d(200.,-10.);
  // Set constraints bounds
  Eigen::Vector2d g_lb_x, g_ub_x;
  Eigen::Vector2d g_lb_u, g_ub_u;
  has_x_cstr_ = false;
  has_u_cstr_ = false;
  if(x_eq){
    g_lb_x = Eigen::Vector2d(0., 0.);
    g_ub_x = Eigen::Vector2d(0., 0.);
    has_x_cstr_ = true;
  }
  if(x_ineq){
    // std::numeric_limits<double>::infinity();
    g_lb_x = Eigen::Vector2d(-0.4, -0.4);
    g_ub_x = Eigen::Vector2d(0.4, 0.4);
    has_x_cstr_ = true;
  }  
  if(u_eq){
    g_lb_u = Eigen::Vector2d(0., 0.);
    g_ub_u = Eigen::Vector2d(0., 0.);
    has_u_cstr_ = true;
  }
  if(u_ineq){
    g_lb_u = Eigen::Vector2d(-1, -1);
    g_ub_u = Eigen::Vector2d(1., 1.);
    has_u_cstr_ = true;
  }
  g_lb_ << g_lb_x, g_lb_u;
  g_ub_ << g_ub_x, g_ub_u;

  
  no_cstr_ = (has_x_cstr_ == false && has_u_cstr_ == false);

  std::cout << "Created DAMPointMass1D" << std::endl;
}


DAMPointMass1D::~DAMPointMass1D() {}


void DAMPointMass1D::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x,
    const Eigen::Ref<const VectorXd>& u) {
  
  // Compute cost 
  costCalc(data, x, u);
  
  // Compute dynamics
  if(isTerminal_ == false){
    data->xout = u + gravity_;
  } else {
    data->xout.setZero();
  }
  
  // Compute constraints
  constraintCalc(data, x, u);
}

void DAMPointMass1D::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x, 
    const Eigen::Ref<const VectorXd>& u) {

  // Compute cost derivatives
  costCalcDiff(data, x, u);
  
  // Compute dynamics derivatives
  data->Fx.setZero();
  if(isTerminal_ == false){
    data->Fu.setIdentity();
  }

  // Compute constraints derivatives
  if(no_cstr_ == false){
    constraintCalcDiff(data, x, u);
  }
}

void DAMPointMass1D::costCalc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                              const Eigen::Ref<const VectorXd>& x, 
                              const Eigen::Ref<const VectorXd>& u){
  if(isTerminal_){
    data->cost = 0.5*pow(x_weights_terminal_[0]*(x[0] - 1.0), 2);
    data->cost += 0.5*pow(x_weights_terminal_[1]*x[1], 2);
  } 
  else {
    data->cost = 0.5*pow((x[0] - 1.0), 2) + pow(x[1],2); 
    data->cost += 0.5*pow(u[0], 2); 
  }
}



void DAMPointMass1D::costCalcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                  const Eigen::Ref<const VectorXd>& x, 
                                  const Eigen::Ref<const VectorXd>& u){
  // Compute derivatives
  if(isTerminal_){
      data->Lx[0] = x_weights_terminal_[0] * (x[0] - 1.);
      data->Lx[1] = x_weights_terminal_[1] * x[1];
      data->Lxx(0,0) = x_weights_terminal_[0];
      data->Lxx(1,1) = x_weights_terminal_[1];
  }
  else{
      data->Lx[0] = x[0] - 1.;
      data->Lx[1] = x[1];
      data->Lu[0] = u[0];
      data->Lu[1] = u[1];
      data->Lxx(0,0) = 1.; 
      data->Lxx(1,1) = 1.; 
      data->Luu(0,0) = 1.;
  }
  data->Luu.setZero();
  data->Lxu.setZero();
}

void DAMPointMass1D::constraintCalc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                    const Eigen::Ref<const VectorXd>& x, 
                                    const Eigen::Ref<const VectorXd>& u){
  // X constraint
  if(has_x_cstr_ == true && isInitial_ == false){
    data->g.head(state_->get_nx());
  }
  if(has_u_cstr_ == true && isTerminal_ == false){
    data->g.tail(nu_) = u;
  }
}

void DAMPointMass1D::constraintCalcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                        const Eigen::Ref<const VectorXd>& x, 
                                        const Eigen::Ref<const VectorXd>& u){
  if(has_x_cstr_ == true && isInitial_ == false){
    data->Gx.topRows(state_->get_nx()).setIdentity();
  }
  if(has_u_cstr_ == true && isTerminal_ == false){
    data->Gu.bottomRows(nu_).setIdentity();
  }
}

boost::shared_ptr<crocoddyl::DifferentialActionDataAbstractTpl<double> >
DAMPointMass1D::createData() {
  return boost::allocate_shared<DADPointMass1D>(Eigen::aligned_allocator<DADPointMass1D>(), this);
}


bool DAMPointMass1D::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<DADPointMass1D> d = boost::dynamic_pointer_cast<DADPointMass1D>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}


void DAMPointMass1D::print(std::ostream& os) const {
  os << "DAMPointMass1D {nq=1, nu=1}";
}

}  // namespace unittest
}  // namespace mim_solvers