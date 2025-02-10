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

#include "constraints.hpp"

#include <crocoddyl/core/utils/exception.hpp>

namespace mim_solvers {
namespace unittest {

const std::vector<XConstraintType::Type> XConstraintType::all(
    XConstraintType::init_all());

std::ostream& operator<<(std::ostream& os, XConstraintType::Type type) {
  switch (type) {
    case XConstraintType::AllEq:
      os << "X-Eq-All";
      break;
    case XConstraintType::AllIneq:
      os << "X-Ineq-All";
      break;
    case XConstraintType::TermEq:
      os << "X-Eq-Term";
      break;
    case XConstraintType::TermIneq:
      os << "X-Ineq-Term";
      break;
    case XConstraintType::None:
      os << "X-None";
      break;
    case XConstraintType::NbXConstraintTypes:
      os << "NbXConstraintTypes";
      break;
    default:
      break;
  }
  return os;
}

const std::vector<UConstraintType::Type> UConstraintType::all(
    UConstraintType::init_all());

std::ostream& operator<<(std::ostream& os, UConstraintType::Type type) {
  switch (type) {
    case UConstraintType::AllEq:
      os << "U-Eq-All";
      break;
    case UConstraintType::AllIneq:
      os << "U-Ineq-All";
      break;
    case UConstraintType::None:
      os << "U-None";
      break;
    case UConstraintType::NbUConstraintTypes:
      os << "NbUConstraintTypes";
      break;
    default:
      break;
  }
  return os;
}

}  // namespace unittest
}  // namespace mim_solvers
