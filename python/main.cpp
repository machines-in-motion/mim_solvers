#include "mim_solvers/python.hpp"

BOOST_PYTHON_MODULE(mim_solvers_pywrap) { 

    namespace bp = boost::python;

    bp::import("crocoddyl");

    mim_solvers::exposeSolverSQP(); 
}
