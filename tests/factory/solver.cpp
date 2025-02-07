///////////////////////////////////////////////////////////////////////////////
// 
// This file is a modified version of the solvers unittests from the Crocoddyl library
// This modified version is used to test our solvers (mim_solvers repository)
// Original file : https://github.com/loco-3d/crocoddyl/blob/devel/unittest/factory/solver.cpp
// 
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "solver.hpp"

#include "mim_solvers/kkt.hpp"
#include "mim_solvers/ddp.hpp"
#include "mim_solvers/fddp.hpp"
#include "mim_solvers/sqp.hpp"
#include "mim_solvers/csqp.hpp"
#ifdef MIM_SOLVERS_WITH_PROXQP
  #include "mim_solvers/csqp_proxqp.hpp"
#endif
#include <crocoddyl/core/utils/exception.hpp>

namespace mim_solvers {
namespace unittest {

const std::vector<SolverTypes::Type> SolverTypes::all(SolverTypes::init_all());

std::ostream& operator<<(std::ostream& os, SolverTypes::Type type) {
  switch (type) {
    case SolverTypes::SolverSQP:
      os << "SolverSQP";
      break;
    case SolverTypes::SolverCSQP:
      os << "SolverCSQP";
      break;
#ifdef MIM_SOLVERS_WITH_PROXQP
    case SolverTypes::SolverPROXQP:
      os << "SolverPROXQP";
      break;
#endif
    case SolverTypes::NbSolverTypes:
      os << "NbSolverTypes";
      break;
    default:
      break;
  }
  return os;
}

SolverFactory::SolverFactory() {}

SolverFactory::~SolverFactory() {}

std::shared_ptr<crocoddyl::SolverAbstract> SolverFactory::create(
    SolverTypes::Type solver_type, 
    ProblemTypes::Type problem_type,
    ModelTypes::Type model_type,
    XConstraintType::Type x_cstr_type,
    UConstraintType::Type u_cstr_type) const {

  std::shared_ptr<crocoddyl::SolverAbstract> solver;
  std::shared_ptr<crocoddyl::ShootingProblem> problem =
      ProblemFactory().create( problem_type, model_type, x_cstr_type, u_cstr_type );

  switch (solver_type) {
    case SolverTypes::SolverSQP:
      if(x_cstr_type != XConstraintType::None || u_cstr_type != UConstraintType::None){
        throw_pretty(__FILE__": Impossible test case : SolverSQP cannot be tested on constrained problem !")
      }
      solver = std::make_shared<mim_solvers::SolverSQP>(problem);
      break;
    case SolverTypes::SolverCSQP:
      solver = std::make_shared<mim_solvers::SolverCSQP>(problem);
      break;
#ifdef MIM_SOLVERS_WITH_PROXQP
    case SolverTypes::SolverPROXQP:
      solver = std::make_shared<mim_solvers::SolverPROXQP>(problem);
      break;
#endif
    default:
      throw_pretty(__FILE__ ": Wrong SolverTypes::Type given");
      break;
  }
  return solver;
}

}  // namespace unittest
}  // namespace mim_solvers
