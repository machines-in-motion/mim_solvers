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

#ifndef MIM_SOLVERS_PROBLEM_FACTORY_HPP_
#define MIM_SOLVERS_PROBLEM_FACTORY_HPP_

#include <crocoddyl/core/optctrl/shooting.hpp>

#include "model.hpp"

namespace mim_solvers {
namespace unittest {

struct ProblemTypes {
  enum Type {
    ShootingProblem,
    // ShootingProblem_Large,
    NbProblemTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbProblemTypes);
    for (int i = 0; i < NbProblemTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ProblemTypes::Type type);

class ProblemFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit ProblemFactory();
  ~ProblemFactory();

  std::shared_ptr<crocoddyl::ShootingProblem> create(
      ProblemTypes::Type problem_type, ModelTypes::Type model_type,
      XConstraintType::Type x_cstr_type,
      UConstraintType::Type u_cstr_type) const;
};

}  // namespace unittest
}  // namespace mim_solvers

#endif  // MIM_SOLVERS_ PROBLEM_FACTORY_HPP_
