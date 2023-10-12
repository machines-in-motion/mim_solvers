#ifndef __mim_solvers_python__
#define __mim_solvers_python__

#include <pinocchio/multibody/fwd.hpp>  // Must be included first!
#include <boost/python.hpp>
// #include <eigenpy/eigenpy.hpp>

#include "mim_solvers/ddp.hpp"
#include "mim_solvers/fddp.hpp"
#include "mim_solvers/sqp.hpp"


namespace mim_solvers{
    void exposeSolverDDP();
    void exposeSolverFDDP();
    void exposeSolverSQP();
} // namespace mim_solvers

#endif
