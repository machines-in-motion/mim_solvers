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

#ifndef MIM_SOLVERS_CONSTRAINTS_FACTORY_HPP_
#define MIM_SOLVERS_CONSTRAINTS_FACTORY_HPP_

#include <ostream>
#include <vector>

namespace mim_solvers {
namespace unittest {

struct XConstraintType {
  enum Type { AllEq, AllIneq, TermEq, TermIneq, None, NbXConstraintTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbXConstraintTypes);
    for (int i = 0; i < NbXConstraintTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, XConstraintType::Type type);

struct UConstraintType {
  enum Type { AllEq, AllIneq, None, NbUConstraintTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbUConstraintTypes);
    for (int i = 0; i < NbUConstraintTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, UConstraintType::Type type);

}  // namespace unittest
}  // namespace mim_solvers

#endif  // MIM_SOLVERS_CONSTRAINTS_FACTORY_HPP_
