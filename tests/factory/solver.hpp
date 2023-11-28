///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, LAAS-CNRS,
//                          New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef MIM_SOLVERS_SOLVER_FACTORY_HPP_
#define MIM_SOLVERS_SOLVER_FACTORY_HPP_

#include "crocoddyl/action.hpp"
#include <crocoddyl/core/solver-base.hpp>

namespace mim_solvers {
namespace unittest {

struct SolverTypes {
  enum Type {
    SolverKKT,
    SolverDDP,
    SolverFDDP,
    SolverSQP,
    SolverCSQP,
    SolverPROXQP,
    NbSolverTypes
  };
  static std::vector<Type> init_all() {
    std::vector<Type> v;
    v.reserve(NbSolverTypes);
    for (int i = 0; i < NbSolverTypes; ++i) {
#ifndef MIM_SOLVERS_WITH_PROXQP
      if ((Type)i == SolverPROXQP) {
        continue;
      }
#endif
      v.push_back((Type)i);
    }
    return v;
  }
  static const std::vector<Type> all;
};

std::ostream& operator<<(std::ostream& os, SolverTypes::Type type);

class SolverFactory {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverFactory();
  ~SolverFactory();

  boost::shared_ptr<crocoddyl::SolverAbstract> create(
      SolverTypes::Type solver_type, ActionModelTypes::Type action_type,
      size_t T) const;
};

}  // namespace unittest
}  // namespace mim_solvers

#endif  // MIM_SOLVERS_SOLVER_FACTORY_HPP_
