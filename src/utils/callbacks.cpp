///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mim_solvers/utils/callbacks.hpp"
#include "mim_solvers/csqp.hpp"
#include "mim_solvers/sqp.hpp"
#include <crocoddyl/core/utils/exception.hpp>
#include <boost/pointer_cast.hpp>

#define BOOST_BIND_NO_PLACEHOLDERS

namespace mim_solvers {

CallbackVerbose::CallbackVerbose(std::string solver_type, int precision)
    : CallbackAbstract(),
      solver_type_(solver_type),
      separator_("  "),
      separator_short_(" ") {
  set_precision(precision);
}

CallbackVerbose::~CallbackVerbose() {}

std::string CallbackVerbose::get_solver_type() const { return solver_type_; }

void CallbackVerbose::set_solver_type(std::string& solver_type) {
    solver_type_ = solver_type;
    update_header();
}

int CallbackVerbose::get_precision() const { return precision_; }

void CallbackVerbose::set_precision(int precision) {
  if (precision < 0) throw_pretty("The precision needs to be at least 0.");
  precision_ = precision;
  update_header();
}

void CallbackVerbose::update_header() {
  auto center_string = [](const std::string& str, int width,
                          bool right_padding = true) {
    const int padding_size = width - static_cast<int>(str.length());
    const int padding_left = padding_size > 0 ? padding_size / 2 : 0;
    const int padding_right =
        padding_size % 2 != 0
            ? padding_left + 1
            : padding_left;  // If the padding is odd, add additional space
    if (right_padding) {
      return std::string(padding_left, ' ') + str +
             std::string(padding_right, ' ');
    } else {
      return std::string(padding_left, ' ') + str;
    }
  };

  header_.clear();
  // Scientific mode requires a column width of 6 + precision
  const int columnwidth = 6 + precision_;
  header_ += "iter" + separator_;
  if (solver_type_ == "CSQP") {
    header_ += center_string("merit", columnwidth) + separator_;
    header_ += center_string("cost", columnwidth) + separator_;
    header_ += center_string("||gaps||", columnwidth) + separator_;
    header_ += center_string("||Constraint||", columnwidth) + separator_;
    header_ += center_string("||(dx,du)||", columnwidth) + separator_;
    header_ += center_string("step", columnwidth) + separator_;
    header_ += center_string("KKT criteria", columnwidth) + separator_;
    header_ += center_string("QP iters", columnwidth);
  } else if (solver_type_ == "SQP"){
    header_ += center_string("merit", columnwidth) + separator_;
    header_ += center_string("cost", columnwidth) + separator_;
    header_ += center_string("||gaps||", columnwidth) + separator_;
    header_ += center_string("step", columnwidth) + separator_;
    header_ += center_string("KKT criteria", columnwidth) + separator_;
  }
  else {
    throw_pretty("The solver type is not supported.");
  }
}

void CallbackVerbose::operator()(SolverAbstract& solver) {
  if (solver.get_iter() % 10 == 0) {
    std::cout << header_ << std::endl;
  }
  auto space_sign = [this](const double value) {
    std::stringstream stream;
    if (value >= 0.) {
      stream << " ";
    } else {
      stream << "-";
    }
    stream << std::scientific << std::setprecision(precision_) << abs(value);
    return stream.str();
  };

  std::cout << std::setw(4) << solver.get_iter() << separator_;                                     // iter
    if (solver_type_ == "CSQP"){
      SolverCSQP& solver_cast = static_cast<SolverCSQP&>(solver);
      if(solver_cast.get_KKT() < solver_cast.get_termination_tolerance()){
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_merit() << separator_;                                            // merit
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_cost() << separator_;                                             // cost
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_gap_norm() << separator_ << separator_;                                         // ||gap||
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_constraint_norm() << separator_;                                  // ||Constraint||
        std::cout << std::scientific << std::setprecision(precision_) 
                    << "       ---- "  << separator_;                                                   // No ||(dx,du)||
        std::cout << std::scientific << std::setprecision(precision_)
                    << "     ---- "  << separator_ << separator_;                                                  // No step
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_KKT() << separator_ << separator_;                                              // KKT criteria
        std::cout << std::scientific << std::setprecision(precision_)
                    << "    -----" << separator_;                                                       // No QP iters     
        }else{
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_merit() << separator_;                                            // merit
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_cost() << separator_;                                             // cost
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_gap_norm() << separator_ << separator_;                                         // ||gap||
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_constraint_norm() << separator_ << separator_ << separator_;                                  // ||Constraint||
        std::cout << std::scientific << std::setprecision(precision_) 
                    << (solver_cast.get_xgrad_norm() + solver_cast.get_ugrad_norm()) / 2  << separator_ << separator_;      // ||(dx,du)||
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_steplength() << separator_ ;                                       // step
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_KKT() << separator_ << separator_ << separator_short_;                                              // KKT criteria
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_qp_iters() << separator_;                                         // QP iters                            
      
        }
    } else if (solver_type_ == "SQP"){
      SolverSQP& solver_cast = static_cast<SolverSQP&>(solver);
      if(solver_cast.get_KKT() < solver_cast.get_termination_tolerance()){
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_merit() << separator_;                                            // merit
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_cost() << separator_;                                             // cost
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_gap_norm() << separator_ << separator_;                                         // ||gap||
        std::cout << std::scientific << std::setprecision(precision_)
                    << "     ---- "  << separator_ << separator_;                                                  // No step
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_KKT() << separator_ << separator_;                                              // KKT criteria
      } else{
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_merit() << separator_;                                            // merit
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_cost() << separator_;                                             // cost
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_gap_norm() << separator_ << separator_;                                         // ||gap||
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_steplength() << separator_ ;                                       // step
        std::cout << std::scientific << std::setprecision(precision_)
                    << solver_cast.get_KKT() << separator_ << separator_ << separator_short_;                                              // KKT criteria
      }
    }

  
  std::cout << std::endl;
  std::cout << std::flush;
}

}  // namespace crocoddyl