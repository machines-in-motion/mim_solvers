///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "mim_solvers/utils/callbacks.hpp"

#include "mim_solvers/python.hpp"

namespace mim_solvers {
namespace bp = boost::python;

void exposeCallbackVerbose() {
  bp::register_ptr_to_python<std::shared_ptr<CallbackAbstract> >();

  bp::class_<CallbackVerbose, bp::bases<mim_solvers::CallbackAbstract> >(
      "CallbackVerbose", "Callback function for printing the solver values.",
      bp::init<bp::optional<int> >(
          bp::args("self", "precision"),
          "Initialize the differential verbose callback.\n\n"
          ":param precision: precision of floating point output (default 3)"))

      .add_property("precision", &CallbackVerbose::get_precision,
                    &CallbackVerbose::set_precision, "precision");
}

}  // namespace mim_solvers
