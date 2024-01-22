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

#include "model.hpp"

// #include "crocoddyl/core/costs/residual.hpp"
// #include "crocoddyl/core/residuals/control.hpp"
// #include "crocoddyl/multibody/residuals/control-gravity.hpp"
// #include "crocoddyl/multibody/residuals/state.hpp"
// // #include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"
// #include "crocoddyl/core/activations/quadratic.hpp"
// #include "crocoddyl/core/costs/cost-sum.hpp"
// #include "crocoddyl/core/utils/exception.hpp"
// #include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
// #include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
// #include "crocoddyl/multibody/residuals/frame-placement.hpp"
// #include "crocoddyl/multibody/residuals/pair-collision.hpp"

namespace mim_solvers {
namespace unittest {

const std::vector<ModelTypes::Type> ModelTypes::all(
    ModelTypes::init_all());

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

boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> ModelFactory::create(
      ModelTypes::Type model_type, 
      XConstraintType::Type x_cstr_type,
      UConstraintType::Type u_cstr_type,
      bool is_terminal) const {
  
  switch(model_type) {
    case ModelTypes::PointMass1D:
      // create point mass 1D DAM
      model = boost::make_shared<DAMPointMass1D>();
      break;
    case ModelTypes::PointMass2D:
      // Create point mass 2D
      model = boost::make_shared<DAMPointMass2D>(is_terminal);
      break;
    default:
      throw_pretty(__FILE__ ": Wrong ModelTypes::Type given");
      break;
  }
  std::size_t ng_x = 0;
  std::size_t ng_u = 0;
  // Create state equality constraint
  if(x_cstr_type==XConstraintType::AllEq || (x_cstr_type==XConstraintType::TermEq && is_terminal==true)){
    ng_x = 2
    model->set_x_eq_cstr();
  }
  // Create state inequality constraint
  if(x_cstr_type==XConstraintType::AllIneq || (x_cstr_type==XConstraintType::TermIneq && is_terminal==true)){
    ng_x = 2
    low, upper
    model->set_x_ineq_cstr();
  }
  // Create control equality constraint
  if(u_cstr_type==UConstraintType::AllEq && is_terminal==false){
    ng_u = 1
    model->set_u_eq_cstr();
  }
  // Create control inequality constraint
  if(u_cstr_type==UConstraintType::AllIneq && is_terminal==false){
    std::cout
    // ng_u = 1
    // model->set_u_ineq_cstr();
  }
  std::size_t ng = ng_x + ng_u;

  return model;
}


// boost::shared_ptr<crocoddyl::CostModelAbstract> create_random_cost(
//     StateModelTypes::Type state_type, std::size_t nu) {
//   static bool once = true;
//   if (once) {
//     srand((unsigned)time(NULL));
//     once = false;
//   }

//   ModelFactory factory;
//   ModelTypes::Type rand_type = static_cast<ModelTypes::Type>(
//       rand() % ModelTypes::NbModelTypes);
//   return factory.create(rand_type, state_type,
//                         ActivationModelTypes::ActivationModelQuad, nu);
// }

}  // namespace unittest
}  // namespace mim_solvers
