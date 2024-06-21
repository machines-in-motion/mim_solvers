///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_UTILS_CALLBACKS_HPP_
#define MIM_SOLVERS_UTILS_CALLBACKS_HPP_

#include <iomanip>
#include <iostream>
#include <crocoddyl/core/solver-base.hpp>

using namespace crocoddyl;
namespace mim_solvers {

class CallbackVerbose : public CallbackAbstract {
 public:
  explicit CallbackVerbose(std::string solver_type = "CSQP", int precision = 3);
  ~CallbackVerbose() override;

  void operator()(SolverAbstract& solver) override;
  
  void set_solver_type(std::string& solver_type);

  std::string get_solver_type() const;

  int get_precision() const;
  void set_precision(int precision);

 private:
  std::string solver_type_;
  int precision_;
  std::string header_;
  std::string separator_;
  std::string separator_short_;

  void update_header();
};

}  // namespace mim_solvers

#endif  // MIM_SOLVERS_UTILS_CALLBACKS_HPP_
