///////////////////////////////////////////////////////////////////////////////
// 
// This file is a modified version of the state model unittests factory from the Crocoddyl library
// This modified version is used for testing purposes only
// Original file : https://github.com/loco-3d/crocoddyl/blob/devel/unittest/factory/state.cpp
// 
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "state.hpp"

#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace mim_solvers {
namespace unittest {
using namespace crocoddyl;

const std::vector<StateModelTypes::Type> StateModelTypes::all(
    StateModelTypes::init_all());

std::ostream& operator<<(std::ostream& os, StateModelTypes::Type type) {
  switch (type) {
    case StateModelTypes::StateVector:
      os << "StateVector";
      break;
    case StateModelTypes::StateMultibody_Hector:
      os << "StateMultibody_Hector";
      break;
    case StateModelTypes::StateMultibody_TalosArm:
      os << "StateMultibody_TalosArm";
      break;
    case StateModelTypes::StateMultibody_HyQ:
      os << "StateMultibody_HyQ";
      break;
    case StateModelTypes::StateMultibody_Talos:
      os << "StateMultibody_Talos";
      break;
    case StateModelTypes::StateMultibody_RandomHumanoid:
      os << "StateMultibody_RandomHumanoid";
      break;
    case StateModelTypes::NbStateModelTypes:
      os << "NbStateModelTypes";
      break;
    default:
      break;
  }
  return os;
}

StateModelFactory::StateModelFactory() {}
StateModelFactory::~StateModelFactory() {}

boost::shared_ptr<crocoddyl::StateAbstract> StateModelFactory::create(
    StateModelTypes::Type state_type) const {
  boost::shared_ptr<pinocchio::Model> model;
  boost::shared_ptr<crocoddyl::StateAbstract> state;
  switch (state_type) {
    case StateModelTypes::StateVector:
      state = boost::make_shared<crocoddyl::StateVector>(80);
      break;
    case StateModelTypes::StateMultibody_Hector:
      model = PinocchioModelFactory(PinocchioModelTypes::Hector).create();
      state = boost::make_shared<crocoddyl::StateMultibody>(model);
      break;
    case StateModelTypes::StateMultibody_TalosArm:
      model = PinocchioModelFactory(PinocchioModelTypes::TalosArm).create();
      state = boost::make_shared<crocoddyl::StateMultibody>(model);
      break;
    case StateModelTypes::StateMultibody_HyQ:
      model = PinocchioModelFactory(PinocchioModelTypes::HyQ).create();
      state = boost::make_shared<crocoddyl::StateMultibody>(model);
      break;
    case StateModelTypes::StateMultibody_Talos:
      model = PinocchioModelFactory(PinocchioModelTypes::Talos).create();
      state = boost::make_shared<crocoddyl::StateMultibody>(model);
      break;
    case StateModelTypes::StateMultibody_RandomHumanoid:
      model =
          PinocchioModelFactory(PinocchioModelTypes::RandomHumanoid).create();
      state = boost::make_shared<crocoddyl::StateMultibody>(model);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong StateModelTypes::Type given");
      break;
  }
  return state;
}

}  // namespace unittest
}  // namespace mim_solvers
