#ifndef __mim_solvers_python__
#define __mim_solvers_python__

#include <boost/python.hpp>

#include "mim_solvers/ddp.hpp"
#include "mim_solvers/fddp.hpp"
#include "mim_solvers/sqp.hpp"
#include "mim_solvers/csqp.hpp"
#include "mim_solvers/csqp_proxqp.hpp"
#include "mim_solvers/csqp_hpipm.hpp"


namespace mim_solvers{
    void exposeSolverDDP();
    void exposeSolverFDDP();
    void exposeSolverSQP();
    void exposeSolverCSQP();
    void exposeSolverPROXQP();
    void exposeSolverHPIPM();
} // namespace mim_solvers

#endif
