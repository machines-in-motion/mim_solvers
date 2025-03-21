///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "point-mass.hpp"

#include "crocoddyl/core/utils/exception.hpp"

namespace mim_solvers {
namespace unittest {

DAMPointMass1D::DAMPointMass1D(const std::size_t ng, const bool x_eq,
                               const bool x_ineq, const bool u_eq,
                               const bool u_ineq, const bool isInitial,
                               const bool isTerminal)
    : DAMBase(std::make_shared<crocoddyl::StateVector>(2), 1, 1, ng) {
  state_ = std::make_shared<crocoddyl::StateVector>(2);

  // Check constraint consistency (safe-guard for bugs in external logic)
  if ((u_eq && isTerminal) || (u_ineq && isTerminal)) {
    throw_pretty(__FILE__
                 ": terminal model cannot have a control constraint !");
  }
  if ((x_eq && isInitial) || (x_ineq && isInitial)) {
    throw_pretty(__FILE__ ": initial model cannot have a state constraint !");
  }
  if ((u_eq && u_ineq) || (x_eq && x_ineq)) {
    throw_pretty(__FILE__ ": test case unsupported (too many constraints) !");
  }
  if (x_eq || x_ineq || u_eq || u_ineq) {
    no_cstr_ = false;
  } else {
    no_cstr_ = true;
  }
  if (no_cstr_ == true && ng != 0) {
    // std::cerr << "ng =" << ng << " is detected." << std::endl;
    // std::cerr << "x_eq =" << x_eq << std::endl;
    // std::cerr << "x_ineq =" << x_ineq << std::endl;
    // std::cerr << "u_eq =" << u_eq << std::endl;
    // std::cerr << "u_ineq =" << u_ineq << std::endl;
    // std::cerr << "isInitial =" << isInitial << std::endl;
    // std::cerr << "isTerminal =" << isTerminal << std::endl;
    throw_pretty(__FILE__ ": no constraint defined so ng must be zero !");
  }
  if (no_cstr_ == false && ng == 0) {
    // std::cerr << "ng =" << ng << " is detected." << std::endl;
    // std::cerr << "x_eq =" << x_eq << std::endl;
    // std::cerr << "x_ineq =" << x_ineq << std::endl;
    // std::cerr << "u_eq =" << u_eq << std::endl;
    // std::cerr << "u_ineq =" << u_ineq << std::endl;
    // std::cerr << "isInitial =" << isInitial << std::endl;
    // std::cerr << "isTerminal =" << isTerminal << std::endl;
    throw_pretty(__FILE__ ": Error: constraint are defined so ng must > 0 !");
  }
  // Model parameters
  isTerminal_ = isTerminal;
  isInitial_ = isInitial;
  gravity_.resize(1);
  gravity_ << -9.81;
  x_weights_terminal_ = Eigen::Vector2d(200., 10.);
  target_.resize(1);
  target_ << 1.;
  has_x_cstr_ = false;
  has_u_cstr_ = false;
  if (no_cstr_ == false) {
    if (x_eq) {
      g_lb_.head(state_->get_nx()) = Eigen::Vector2d(0., 0.);
      g_ub_.head(state_->get_nx()) = Eigen::Vector2d(0., 0.);
      has_x_cstr_ = true;
    }
    if (x_ineq) {
      g_lb_.head(state_->get_nx()) = Eigen::Vector2d(0., 0.);
      g_ub_.head(state_->get_nx()) = Eigen::Vector2d(0.4, 0.4);
      has_x_cstr_ = true;
    }
    if (u_eq) {
      g_lb_.tail(nu_) = Eigen::VectorXd::Ones(nu_) * 0.34;
      g_ub_.tail(nu_) = Eigen::VectorXd::Ones(nu_) * 10;
      has_u_cstr_ = true;
    }
    if (u_ineq) {
      g_lb_.tail(nu_) = Eigen::VectorXd::Zero(nu_);
      g_ub_.tail(nu_) = 10 * Eigen::VectorXd::Ones(nu_);
      has_u_cstr_ = true;
    }
  }
}

DAMPointMass1D::~DAMPointMass1D() {}

void DAMPointMass1D::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x, const Eigen::Ref<const VectorXd>& u) {
  // Compute running cost
  data->cost = 0.5 * pow((x[0] - target_[0]), 2) + pow(x[1], 2);
  data->cost += 0.5 * pow(u[0], 2);
  // Compute dynamics
  data->xout = u + gravity_;
  // Compute constraints
  // X constraint
  if (has_x_cstr_ == true && isInitial_ == false) {
    data->g.head(state_->get_nx()) = x;
  }
  // U constraint
  if (has_u_cstr_ == true && isTerminal_ == false) {
    data->g.tail(nu_) = u;
  }
}

void DAMPointMass1D::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x) {
  // Compute terminal cost
  data->cost = 0.5 * pow(x_weights_terminal_[0] * (x[0] - target_[0]), 2);
  data->cost += 0.5 * pow(x_weights_terminal_[1] * x[1], 2);
  // Compute dynamics
  data->xout.setZero();
  // Compute constraints
  if (no_cstr_ == false) {
    if (has_x_cstr_ == true && isInitial_ == false) {
      data->g.head(state_->get_nx()) = x;
    }
  }
}

void DAMPointMass1D::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x, const Eigen::Ref<const VectorXd>& u) {
  // Compute running cost derivatives
  data->Lx[0] = x[0] - target_[0];
  data->Lx[1] = x[1];
  data->Lu[0] = u[0];
  data->Lxx.setIdentity();
  data->Luu.setIdentity();
  // Compute dynamics derivatives
  data->Fx.setZero();
  data->Fu.setIdentity();
  // Compute constraints derivatives
  if (no_cstr_ == false) {
    if (has_x_cstr_ == true && isInitial_ == false) {
      data->Gx.topRows(state_->get_nx()).setIdentity();
    }
    if (has_u_cstr_ == true && isTerminal_ == false) {
      data->Gu.bottomRows(nu_).setIdentity();
    }
  }
}

void DAMPointMass1D::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x) {
  // Compute terminal cost derivatives
  data->Lx[0] = x_weights_terminal_[0] * (x[0] - target_[0]);
  data->Lx[1] = x_weights_terminal_[1] * x[1];
  data->Lxx(0, 0) = x_weights_terminal_[0];
  data->Lxx(1, 1) = x_weights_terminal_[1];
  // Compute dynamics derivatives
  data->Fx.setZero();
  // Compute constraints derivatives
  if (no_cstr_ == false) {
    if (has_x_cstr_ == true && isInitial_ == false) {
      data->Gx.topRows(state_->get_nx()).setIdentity();
    }
  }
}

std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>
DAMPointMass1D::createData() {
  return std::allocate_shared<DADPointMass1D>(
      Eigen::aligned_allocator<DADPointMass1D>(), this);
}

bool DAMPointMass1D::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<DADPointMass1D> d =
      std::dynamic_pointer_cast<DADPointMass1D>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

void DAMPointMass1D::print(std::ostream& os) const {
  os << "DAMPointMass1D {nx=2, nu=1}";
}

DAMPointMass2D::DAMPointMass2D(const std::size_t ng, const bool x_eq,
                               const bool x_ineq, const bool u_eq,
                               const bool u_ineq, const bool isInitial,
                               const bool isTerminal)
    : DAMBase(std::make_shared<crocoddyl::StateVector>(4), 2, 1, ng) {
  state_ = std::make_shared<crocoddyl::StateVector>(4);

  // Check constraint consistency (safe-guard for bugs in external logic)
  if ((u_eq && isTerminal) || (u_ineq && isTerminal)) {
    throw_pretty(__FILE__
                 ": terminal model cannot have a control constraint !");
  }
  if ((x_eq && isInitial) || (x_ineq && isInitial)) {
    throw_pretty(__FILE__ ": initial model cannot have a state constraint !");
  }
  if ((u_eq && u_ineq) || (x_eq && x_ineq)) {
    throw_pretty(__FILE__ ": test case unsupported (too many constraints) !");
  }
  if (x_eq || x_ineq || u_eq || u_ineq) {
    no_cstr_ = false;
  } else {
    no_cstr_ = true;
  }
  if (no_cstr_ == true && ng != 0) {
    // std::cerr << "ng =" << ng << " is detected." << std::endl;
    // std::cerr << "x_eq =" << x_eq << std::endl;
    // std::cerr << "x_ineq =" << x_ineq << std::endl;
    // std::cerr << "u_eq =" << u_eq << std::endl;
    // std::cerr << "u_ineq =" << u_ineq << std::endl;
    // std::cerr << "isInitial =" << isInitial << std::endl;
    // std::cerr << "isTerminal =" << isTerminal << std::endl;
    throw_pretty(__FILE__ ": no constraint defined so ng must be zero !");
  }
  if (no_cstr_ == false && ng == 0) {
    // std::cerr << "ng =" << ng << " is detected." << std::endl;
    // std::cerr << "x_eq =" << x_eq << std::endl;
    // std::cerr << "x_ineq =" << x_ineq << std::endl;
    // std::cerr << "u_eq =" << u_eq << std::endl;
    // std::cerr << "u_ineq =" << u_ineq << std::endl;
    // std::cerr << "isInitial =" << isInitial << std::endl;
    // std::cerr << "isTerminal =" << isTerminal << std::endl;
    throw_pretty(__FILE__ ": Error: constraint are defined so ng must > 0 !");
  }

  // Cost and dynamics parameters
  isTerminal_ = isTerminal;
  isInitial_ = isInitial;
  gravity_ = Eigen::Vector2d(0, -9.81);
  x_weights_terminal_ = Eigen::VectorXd::Zero(4);
  x_weights_terminal_ << 200., 200., 10., 10.;
  target_.resize(2);
  target_ << 1., 0.;

  // Set constraints bounds if any
  has_x_cstr_ = false;
  has_u_cstr_ = false;
  if (no_cstr_ == false) {
    if (x_eq) {
      g_lb_.head(state_->get_nx()) << 0., 0., 0., 0.;
      g_ub_.head(state_->get_nx()) << 0., 0., 0., 0.;
      has_x_cstr_ = true;
    }
    if (x_ineq) {
      g_lb_.head(state_->get_nx()) << -0.4, -0.4, -0.4, -0.4;
      g_ub_.head(state_->get_nx()) << 0.4, 0.4, 0.4, 0.4;
      has_x_cstr_ = true;
    }
    if (u_eq) {
      g_lb_.tail(nu_) = -gravity_;  // 0*Eigen::VectorXd::Ones(nu_);
      g_ub_.tail(nu_) = -gravity_;  // 10*Eigen::VectorXd::Ones(nu_);
      has_u_cstr_ = true;
    }
    if (u_ineq) {
      g_lb_.tail(nu_) = Eigen::VectorXd::Zero(nu_);
      g_ub_.tail(nu_) = 10 * Eigen::VectorXd::Ones(nu_);
      has_u_cstr_ = true;
    }
  }
  // if(no_cstr_ == true){
  //   std::cerr << "g_lb_ =" << g_lb_ << std::endl;
  //   std::cerr << "g_ub_ =" << g_ub_ << std::endl;
  // }
}

DAMPointMass2D::~DAMPointMass2D() {}

void DAMPointMass2D::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x, const Eigen::Ref<const VectorXd>& u) {
  // Compute running cost
  data->cost =
      pow((x[0] - target_[0]), 2) + pow(x[1], 2) + pow(x[2], 2) + pow(x[3], 2);
  data->cost += pow(u[0], 2) + pow(u[1], 2);
  data->cost *= 0.5;
  // Compute dynamics
  data->xout = u + gravity_;
  // Compute constraints
  if (no_cstr_ == false) {
    if (has_x_cstr_ == true && isInitial_ == false) {
      data->g.head(state_->get_nx()) = x;
    }
    if (has_u_cstr_ == true && isTerminal_ == false) {
      data->g.tail(nu_) = u;
    }
  }
}

void DAMPointMass2D::calc(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x) {
  // Compute terminal cost
  data->cost = pow(x_weights_terminal_[0] * (x[0] - target_[0]), 2);
  data->cost += pow(x_weights_terminal_[1] * x[1], 2);
  data->cost += pow(x_weights_terminal_[2] * x[2], 2);
  data->cost += pow(x_weights_terminal_[3] * x[3], 2);
  data->cost *= 0.5;
  // Compute dynamics
  data->xout.setZero();
  // Compute constraints
  if (no_cstr_ == false) {
    if (has_x_cstr_ == true && isInitial_ == false) {
      data->g.head(state_->get_nx()) = x;
    }
  }
}

void DAMPointMass2D::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x, const Eigen::Ref<const VectorXd>& u) {
  // Compute running cost derivatives
  data->Lx[0] = x[0] - target_[0];
  data->Lx[1] = x[1];
  data->Lx[2] = x[2];
  data->Lx[3] = x[3];
  data->Lu[0] = u[0];
  data->Lu[1] = u[1];
  data->Lxx.setIdentity();
  data->Luu.setIdentity();
  // Compute dynamics derivatives
  data->Fx.setZero();
  data->Fu.setIdentity();
  // Compute constraints derivatives
  if (no_cstr_ == false) {
    if (has_x_cstr_ == true && isInitial_ == false) {
      data->Gx.topRows(state_->get_nx()).setIdentity();
    }
    if (has_u_cstr_ == true && isTerminal_ == false) {
      data->Gu.bottomRows(nu_).setIdentity();
    }
  }
}

void DAMPointMass2D::calcDiff(
    const std::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXd>& x) {
  // Compute terminal cost derivatives
  data->Lx[0] = x_weights_terminal_[0] * (x[0] - target_[0]);
  data->Lx[1] = x_weights_terminal_[1] * x[1];
  data->Lx[2] = x_weights_terminal_[2] * x[2];
  data->Lx[3] = x_weights_terminal_[3] * x[3];
  data->Lxx(0, 0) = x_weights_terminal_[0];
  data->Lxx(1, 1) = x_weights_terminal_[1];
  data->Lxx(2, 2) = x_weights_terminal_[2];
  data->Lxx(3, 3) = x_weights_terminal_[3];
  // Compute dynamics derivatives
  data->Fx.setZero();
  data->Fu.setIdentity();
  // Compute constraints derivatives
  if (no_cstr_ == false) {
    if (has_x_cstr_ == true && isInitial_ == false) {
      data->Gx.topRows(state_->get_nx()).setIdentity();
      data->Gu.setZero();
    }
  }
}

std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>
DAMPointMass2D::createData() {
  return std::allocate_shared<DADPointMass2D>(
      Eigen::aligned_allocator<DADPointMass2D>(), this);
}

bool DAMPointMass2D::checkData(
    const std::shared_ptr<DifferentialActionDataAbstract>& data) {
  std::shared_ptr<DADPointMass2D> d =
      std::dynamic_pointer_cast<DADPointMass2D>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

void DAMPointMass2D::print(std::ostream& os) const {
  os << "DAMPointMass2D {nx=4, nu=2}";
}

}  // namespace unittest
}  // namespace mim_solvers
