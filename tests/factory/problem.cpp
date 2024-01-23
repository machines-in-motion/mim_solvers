///////////////////////////////////////////////////////////////////////////////
// 
// This file is a modified version of the cost model unittests factory from the Crocoddyl library
// This modified version is used for testing purposes only
// Original file : https://github.com/loco-3d/crocoddyl/blob/devel/unittest/factory/cost.cpp
// 
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "problem.hpp"

#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/core/integrator/euler.hpp>

#include <vector>

namespace mim_solvers {
namespace unittest {

const std::vector<ProblemTypes::Type> ProblemTypes::all(
    ProblemTypes::init_all());

std::ostream& operator<<(std::ostream& os, ProblemTypes::Type type) {
  switch (type) {
    case ProblemTypes::ShootingProblem:
      os << "ShootingProblem";
      break;
    // case ProblemTypes::ShootingProblem_Large:
    //   os << "ShootingProblem_Large";
    //   break;
    case ProblemTypes::NbProblemTypes:
      os << "NbProblemTypes";
      break;
    default:
      break;
  }
  return os;
}


ProblemFactory::ProblemFactory() {}
ProblemFactory::~ProblemFactory() {}

boost::shared_ptr<crocoddyl::ShootingProblem> ProblemFactory::create(
      ProblemTypes::Type problem_type, 
      ModelTypes::Type model_type, 
      XConstraintType::Type x_cstr_type,
      UConstraintType::Type u_cstr_type) const {
  
  boost::shared_ptr<crocoddyl::ShootingProblem> problem;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> IAMs;
  boost::shared_ptr<crocoddyl::ActionModelAbstract> IAMt;
  
  // Pick initial state depending on the model
  Eigen::VectorXd x0;
  switch (model_type)
  {
  case ModelTypes::PointMass1D:
    x0 = Eigen::VectorXd::Zero(2);
    break;
  case ModelTypes::PointMass2D:
    x0 = Eigen::VectorXd::Zero(4);
    break;
  default:
    break;
  }
  
  // Select OCP parameters
  std::size_t T;
  double DT;
  switch (problem_type)
  {
  case ProblemTypes::ShootingProblem:
    T = 10;
    DT = 0.1;
    break;
//   case ProblemTypes::ShootingProblem_Large:
//     T = 100;
//     DT = 0.1;
//     break;
  default:
    throw_pretty(__FILE__": unknown problem type !");
    break;
  }

  // Construct OCP 
  bool isInitial = true;
  for(std::size_t t=0; t<T; t++){
      if(t > 0){
          isInitial = false;
      }
      IAMs.push_back(boost::make_shared<crocoddyl::IntegratedActionModelEuler>(ModelFactory().create(model_type, x_cstr_type, u_cstr_type, isInitial, false), DT));
  }
  IAMt = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(ModelFactory().create(model_type, x_cstr_type, u_cstr_type, isInitial, false), 0.);
  problem = boost::make_shared<crocoddyl::ShootingProblem>(x0, IAMs, IAMt);
  
  return problem;
}

}  // namespace unittest
}  // namespace mim_solvers
