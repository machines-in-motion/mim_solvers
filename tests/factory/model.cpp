///////////////////////////////////////////////////////////////////////////////
//
// This file is a modified version of the cost model unittests factory from the
// Crocoddyl library This modified version is used for testing purposes only
// Original file :
// https://github.com/loco-3d/crocoddyl/blob/devel/unittest/factory/cost.cpp
//
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "model.hpp"

#include "crocoddyl/core/utils/exception.hpp"

namespace mim_solvers {
namespace unittest {

const std::vector<ModelTypes::Type> ModelTypes::all(ModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, ModelTypes::Type type) {
  switch (type) {
    case ModelTypes::PointMass1D:
      os << "PointMass1D";
      break;
    case ModelTypes::PointMass2D:
      os << "PointMass2D";
      break;
    case ModelTypes::NbModelTypes:
      os << "NbModelTypes";
      break;
    default:
      break;
  }
  return os;
}

ModelFactory::ModelFactory() {}
ModelFactory::~ModelFactory() {}

std::shared_ptr<crocoddyl::DifferentialActionModelAbstract>
ModelFactory::create(ModelTypes::Type model_type,
                     XConstraintType::Type x_cstr_type,
                     UConstraintType::Type u_cstr_type, bool isInitial,
                     bool isTerminal) const {
  std::shared_ptr<crocoddyl::DifferentialActionModelAbstract> model;

  bool x_eq = false;
  bool x_ineq = false;
  bool u_eq = false;
  bool u_ineq = false;

  // state equality constraint
  if ((x_cstr_type == XConstraintType::AllEq && isInitial == false) ||
      (x_cstr_type == XConstraintType::TermEq && isTerminal == true)) {
    x_eq = true;
  }
  // state inequality constraint
  if ((x_cstr_type == XConstraintType::AllIneq && isInitial == false) ||
      (x_cstr_type == XConstraintType::TermIneq && isTerminal == true)) {
    x_ineq = true;
  }
  // control equality constraint
  if (u_cstr_type == UConstraintType::AllEq && isTerminal == false) {
    u_eq = true;
  }
  // control inequality constraint
  if (u_cstr_type == UConstraintType::AllIneq && isTerminal == false) {
    u_ineq = true;
  }

  std::size_t ng_x = 0;
  std::size_t ng_u = 0;
  std::size_t ng = 0;
  switch (model_type) {
    case ModelTypes::PointMass1D:
      if (x_eq || x_ineq) {
        ng_x = 2;
      }
      if (u_eq || u_ineq) {
        ng_u = 1;
      }
      if (isTerminal) {
        ng = ng_x;
      } else if (isInitial) {
        ng = ng_u;
      } else {
        ng = ng_x + ng_u;
      }
      model = std::make_shared<DAMPointMass1D>(ng, x_eq, x_ineq, u_eq, u_ineq,
                                               isInitial, isTerminal);
      break;
    case ModelTypes::PointMass2D:
      if (x_eq || x_ineq) {
        ng_x = 4;
      }
      if (u_eq || u_ineq) {
        ng_u = 2;
      }
      if (isTerminal) {
        ng = ng_x;
      } else if (isInitial) {
        ng = ng_u;
      } else {
        ng = ng_x + ng_u;
      }
      model = std::make_shared<DAMPointMass2D>(ng, x_eq, x_ineq, u_eq, u_ineq,
                                               isInitial, isTerminal);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ModelTypes::Type given");
      break;
  }
  return model;
}

}  // namespace unittest
}  // namespace mim_solvers
