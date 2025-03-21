///////////////////////////////////////////////////////////////////////////////
//
// This file is a modified version of the cost model unittests factory from the
// Crocoddyl library This modified version is used for testing purposes only
// Original file :
// https://github.com/loco-3d/crocoddyl/blob/devel/unittest/factory/cost.hpp
//
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_POINT_MASS_HPP_
#define MIM_SOLVERS_POINT_MASS_HPP_

#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/states/euclidean.hpp>

#include "crocoddyl/core/fwd.hpp"

namespace mim_solvers {
namespace unittest {

struct DADPointMass1D : public crocoddyl::DifferentialActionDataAbstract {
  typedef crocoddyl::DifferentialActionDataAbstract DADBase;
  typedef typename Eigen::VectorXd VectorXd;

  template <class DAModel>
  explicit DADPointMass1D(DAModel* const model) : DADBase(model) {}

  using DADBase::cost;
  using DADBase::Fu;
  using DADBase::Fx;
  using DADBase::g;
  using DADBase::Gu;
  using DADBase::Gx;
  using DADBase::Lu;
  using DADBase::Luu;
  using DADBase::Lx;
  using DADBase::Lxu;
  using DADBase::Lxx;
  using DADBase::r;
  using DADBase::xout;
  // (h,Hx,Hu) not used because our mim_solvers treat equalities as inequalities
  // with lb=ub
};

class DAMPointMass1D : public crocoddyl::DifferentialActionModelAbstract {
 public:
  typedef crocoddyl::DifferentialActionModelAbstract DAMBase;
  typedef typename Eigen::VectorXd VectorXd;

  // Constructor
  DAMPointMass1D(const std::size_t ng, const bool x_eq, const bool x_ineq,
                 const bool u_eq, const bool u_ineq, const bool isInitial,
                 const bool isTerminal = false);

  // Destructor
  virtual ~DAMPointMass1D();

  // Cost & dynamics
  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const VectorXd>& x,
            const Eigen::Ref<const VectorXd>& u);
  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const VectorXd>& x);
  void calcDiff(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const VectorXd>& x,
                const Eigen::Ref<const VectorXd>& u);
  void calcDiff(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const VectorXd>& x);

  virtual std::shared_ptr<DifferentialActionDataAbstract> createData();
  virtual bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data);
  virtual void print(std::ostream& os) const;

 protected:
  using DAMBase::ng_;  //!< Number of inequality constraints
  using DAMBase::nh_;  //!< Number of equality constraints
  using DAMBase::nr_;  //!< Dimension of the cost residual
  using DAMBase::nu_;  //!< Control dimension
  VectorXd gravity_;
  VectorXd x_weights_terminal_;
  VectorXd target_;

  std::shared_ptr<StateAbstract> state_;  //!< Model of the state
  using DAMBase::g_lb_;  //!< Lower bound of the inequality constraints
  using DAMBase::g_ub_;  //!< Lower bound of the inequality constraints
  // VectorXs u_lb_;            //!< Lower control limits
  // VectorXs u_ub_;            //!< Upper control limits
  bool isTerminal_;
  bool isInitial_;
  bool has_x_cstr_;
  bool has_u_cstr_;
  bool no_cstr_;
};

struct DADPointMass2D : public crocoddyl::DifferentialActionDataAbstract {
  typedef crocoddyl::DifferentialActionDataAbstract DADBase;
  typedef typename Eigen::VectorXd VectorXd;

  template <class DAModel>
  explicit DADPointMass2D(DAModel* const model) : DADBase(model) {}

  using DADBase::cost;
  using DADBase::Fu;
  using DADBase::Fx;
  using DADBase::g;
  using DADBase::Gu;
  using DADBase::Gx;
  using DADBase::Lu;
  using DADBase::Luu;
  using DADBase::Lx;
  using DADBase::Lxu;
  using DADBase::Lxx;
  using DADBase::r;
  using DADBase::xout;
  // (h,Hx,Hu) not used because our mim_solvers treat equalities as inequalities
  // with lb=ub
};

class DAMPointMass2D : public crocoddyl::DifferentialActionModelAbstract {
 public:
  typedef crocoddyl::DifferentialActionModelAbstract DAMBase;
  typedef typename Eigen::VectorXd VectorXd;

  // Constructor
  DAMPointMass2D(const std::size_t ng, const bool x_eq, const bool x_ineq,
                 const bool u_eq, const bool u_ineq, const bool isInitial,
                 const bool isTerminal = false);

  // Destructor
  virtual ~DAMPointMass2D();

  // Cost & dynamics
  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const VectorXd>& x,
            const Eigen::Ref<const VectorXd>& u);
  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const VectorXd>& x);
  void calcDiff(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const VectorXd>& x,
                const Eigen::Ref<const VectorXd>& u);
  void calcDiff(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const VectorXd>& x);

  virtual std::shared_ptr<DifferentialActionDataAbstract> createData();
  virtual bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data);
  virtual void print(std::ostream& os) const;

 protected:
  using DAMBase::ng_;  //!< Number of inequality constraints
  using DAMBase::nh_;  //!< Number of equality constraints
  using DAMBase::nr_;  //!< Dimension of the cost residual
  using DAMBase::nu_;  //!< Control dimension
  VectorXd gravity_;
  VectorXd x_weights_terminal_;
  VectorXd target_;

  std::shared_ptr<StateAbstract> state_;  //!< Model of the state
  using DAMBase::g_lb_;  //!< Lower bound of the inequality constraints
  using DAMBase::g_ub_;  //!< Lower bound of the inequality constraints
  // VectorXs u_lb_;            //!< Lower control limits
  // VectorXs u_ub_;            //!< Upper control limits
  bool isTerminal_;
  bool isInitial_;
  bool has_x_cstr_;
  bool has_u_cstr_;
  bool no_cstr_;
};

}  // namespace unittest
}  // namespace mim_solvers

#endif  // MIM_SOLVERS_POINT_MASS_HPP_
