#include "mim_solvers/python.hpp"

BOOST_PYTHON_MODULE(mim_solvers_pywrap) { 

    // namespace bp = boost::python;

    // bp::import("sobec");

    mim_solvers::exposeSolverSQP(); 
}
