///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mim_solvers/utils/callbacks.hpp"
#include <crocoddyl/core/utils/callbacks.hpp>
#include "mim_solvers/python.hpp"

namespace mim_solvers {
namespace bp = boost::python;

void exposeCallbacks() {
  bp::register_ptr_to_python<boost::shared_ptr<CallbackVerbose> >();

  bp::class_<CallbackVerbose, bp::bases<crocoddyl::CallbackAbstract> >(
      "CallbackVerbose", "Callback function for printing the solver values.",
      bp::init<bp::optional<std::string, int> >(
          bp::args("self", "solver_type", "precision"),
          "Initialize the differential verbose callback.\n\n"
          ":param solver_type: solver type (default 'CSQP')\n"
          ":param precision: precision of floating point output (default 3)"))
      .def("__call__", &CallbackVerbose::operator(), bp::args("self", "solver"),
           "Run the callback function given a solver.\n\n"
           ":param solver: solver to be diagnostic")
      .add_property("solver_type", &CallbackVerbose::get_solver_type,
                    &CallbackVerbose::set_solver_type, "solver type")
      .add_property("precision", &CallbackVerbose::get_precision,
                    &CallbackVerbose::set_precision, "precision");
}

}  // namespace crocoddyl
