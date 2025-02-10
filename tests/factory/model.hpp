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

#ifndef MIM_SOLVERS_MODEL_FACTORY_HPP_
#define MIM_SOLVERS_MODEL_FACTORY_HPP_

#include "constraints.hpp"
#include "point-mass.hpp"

namespace mim_solvers {
namespace unittest {

struct ModelTypes {
  enum Type { PointMass1D, PointMass2D, NbModelTypes };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbModelTypes);
    for (int i = 0; i < NbModelTypes; ++i) {
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, ModelTypes::Type type);

class ModelFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // typedef crocoddyl::MathBaseTpl<double> MathBase;
  // typedef typename MathBase::Vector6s Vector6d;

  explicit ModelFactory();
  ~ModelFactory();

  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> create(
      ModelTypes::Type model_type, XConstraintType::Type x_cstr_type,
      UConstraintType::Type u_cstr_type, bool isInitial, bool isTerminal) const;
};

// std::shared_ptr<crocoddyl::CostModelAbstract> create_random_cost(
//     StateModelTypes::Type state_type,
//     std::size_t nu = std::numeric_limits<std::size_t>::max());
}  // namespace unittest
}  // namespace mim_solvers

#endif  // MIM_SOLVERS_COST_FACTORY_HPP_
