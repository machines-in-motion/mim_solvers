///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (C) 2023, New York University
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef __mim_solvers_python__
#define __mim_solvers_python__

#include <boost/python.hpp>

#include "mim_solvers/csqp.hpp"
#include "mim_solvers/ddp.hpp"
#include "mim_solvers/fddp.hpp"
#include "mim_solvers/sqp.hpp"
#include "mim_solvers/utils/callbacks.hpp"

#ifdef MIM_SOLVERS_WITH_PROXQP
#include "mim_solvers/csqp_proxqp.hpp"
#endif

namespace mim_solvers {
void exposeCallbackAbstract();
void exposeCallbackVerbose();
void exposeSolverDDP();
void exposeSolverFDDP();
void exposeSolverSQP();
void exposeSolverCSQP();
#ifdef MIM_SOLVERS_WITH_PROXQP
void exposeSolverPROXQP();
#endif
}  // namespace mim_solvers

#endif
