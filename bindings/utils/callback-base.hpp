#ifndef BINDINGS_PYTHON_MIM_SOLVERS_CORE_CALLBACK_BASE_HPP_
#define BINDINGS_PYTHON_MIM_SOLVERS_CORE_CALLBACK_BASE_HPP_

#include <boost/python.hpp>
#include <crocoddyl/core/solver-base.hpp>
#include <memory>
#include <vector>

#include "mim_solvers/utils/callbacks.hpp"

namespace mim_solvers {
namespace bp = boost::python;

class CallbackAbstract_wrap : public CallbackAbstract,
                              public bp::wrapper<CallbackAbstract> {
 public:
  CallbackAbstract_wrap()
      : CallbackAbstract(), bp::wrapper<CallbackAbstract>() {}
  ~CallbackAbstract_wrap() {}

  void operator()(crocoddyl::SolverAbstract& solver) override {
    return bp::call<void>(this->get_override("__call__").ptr(),
                          boost::ref(solver));
  }

  void operator()(crocoddyl::SolverAbstract& solver,
                  std::string solver_type) override {
    return bp::call<void>(this->get_override("__call__").ptr(),
                          boost::ref(solver), solver_type);
  }
};

}  // namespace mim_solvers

#endif  // BINDINGS_PYTHON_MIM_SOLVERS_CORE_CALLBACK_BASE_HPP_
