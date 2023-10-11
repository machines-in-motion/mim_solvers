#ifndef __mim_solvers_python__
#define __mim_solvers_python__

#include <pinocchio/multibody/fwd.hpp>  // Must be included first!
#include <boost/python.hpp>

#include "mim_solvers/sqp.hpp"


namespace mim_solvers{
    void exposeSolverSQP();
} // namespace mim_solvers

#endif
