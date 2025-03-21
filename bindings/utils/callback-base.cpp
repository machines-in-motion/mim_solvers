// TO be moved to mim_solvers::SolverAbstract
#include "callback-base.hpp"

#include <eigenpy/std-vector.hpp>

#include "copyable.hpp"
#include "mim_solvers/python.hpp"
#include "mim_solvers/utils/callbacks.hpp"
namespace mim_solvers {
namespace bp = boost::python;

void exposeCallbackAbstract() {
  typedef std::shared_ptr<CallbackAbstract> CallbackAbstractPtr;
  eigenpy::StdVectorPythonVisitor<std::vector<CallbackAbstractPtr>,
                                  true>::expose("StdVec_Callback");

  void (CallbackAbstract_wrap::*callback_operator1)(
      crocoddyl::SolverAbstract&) = &CallbackAbstract_wrap::operator();
  void (CallbackAbstract_wrap::*callback_operator2)(crocoddyl::SolverAbstract&,
                                                    std::string) =
      &CallbackAbstract_wrap::operator();

  bp::class_<CallbackAbstract_wrap, boost::noncopyable>(
      "CallbackAbstract",
      "Abstract class for solver callbacks.\n\n"
      "A callback is used to diagnostic the behaviour of our solver in each "
      "iteration of it.\n"
      "For instance, it can be used to print values, record data or display "
      "motions")
      .def("__call__", bp::pure_virtual(callback_operator1),
           bp::args("self", "solver"),
           "Run the callback function given a solver.\n\n"
           ":param solver: solver to be diagnostic")

      .def("__call__", bp::pure_virtual(callback_operator2),
           bp::args("self", "solver", "solver_type"),
           "Run the callback function given a solver.\n\n"
           ":param solver: solver to be diagnostic")

      .def(CopyableVisitor<CallbackAbstract_wrap>());
}

}  // namespace mim_solvers
