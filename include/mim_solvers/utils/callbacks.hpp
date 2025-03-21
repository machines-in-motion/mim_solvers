///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_UTILS_CALLBACKS_HPP_
#define MIM_SOLVERS_UTILS_CALLBACKS_HPP_

#include <crocoddyl/core/solver-base.hpp>
#include <iomanip>
#include <iostream>

namespace mim_solvers {

/**
 * @brief Abstract class for solver callbacks
 *
 * A callback is used to diagnostic the behaviour of our solver in each
 * iteration of it. For instance, it can be used to print values, record data or
 * display motions.
 */
class CallbackAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the callback function
   */
  CallbackAbstract() {}
  virtual ~CallbackAbstract() {}

  /**
   * @brief Run the callback function given a solver
   *
   * @param[in]  solver solver to be diagnostic
   */
  virtual void operator()(crocoddyl::SolverAbstract& solver) = 0;
  virtual void operator()(crocoddyl::SolverAbstract& solver,
                          std::string solver_type) = 0;
};

class CallbackVerbose : public CallbackAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit CallbackVerbose(int precision = 3);
  ~CallbackVerbose() override;

  void operator()(crocoddyl::SolverAbstract& solver) override;
  void operator()(crocoddyl::SolverAbstract& solver,
                  std::string solver_type) override;

  int get_precision() const;
  void set_precision(int precision);

 private:
  int precision_;
  std::string header_;
  std::string separator_;
  std::string separator_short_;

  void update_header(const std::string solver_type);
};

}  // namespace mim_solvers

#endif  // MIM_SOLVERS_UTILS_CALLBACKS_HPP_
