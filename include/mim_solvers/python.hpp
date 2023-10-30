#ifndef __mim_solvers_python__
#define __mim_solvers_python__

#include <boost/python.hpp>

#include "mim_solvers/ddp.hpp"
#include "mim_solvers/fddp.hpp"
#include "mim_solvers/sqp.hpp"


namespace mim_solvers{
    void exposeSolverDDP();
    void exposeSolverFDDP();
    void exposeSolverSQP();
    void exposeSolverCSQP();
} // namespace mim_solvers

#endif
