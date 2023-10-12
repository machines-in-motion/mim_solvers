#include "mim_solvers/python.hpp"
// #include <eigenpy/eigenpy.hpp>
BOOST_PYTHON_MODULE(mim_solvers_pywrap) { 

    namespace bp = boost::python;

    bp::import("crocoddyl");
    // // Enabling eigenpy support, i.e. numpy/eigen compatibility.
    // eigenpy::enableEigenPy();
    // eigenpy::enableEigenPySpecific<Eigen::VectorXi>();
    mim_solvers::exposeSolverDDP(); 
    mim_solvers::exposeSolverFDDP(); 
    mim_solvers::exposeSolverSQP(); 
}
