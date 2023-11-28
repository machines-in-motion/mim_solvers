# mim_solvers
Implementation of numerical solvers used in the Machines in Motion Laboratory. 
In particular, the Sequential Quadratic Programming (SQP) solver decribed in [arxiv] that solves efficiently nonlinear constrained OCPs.

All solvers are implemented by using [Crocoddyl](https://github.com/loco-3d/crocoddyl/tree/devel) (v2.0) as the base software. 
Consquently, Crocoddyl users can use our efficient solvers while constructing their OCPs using the same API they are used to. 
The default solvers of Crocoddyl are also re-implemented for benchmarking purposes (namely DDP and FDDP) but with modified termination criteria and line-search.

Examples of how to use the solvers can be found in the `examples` directory.

# Dependencies
- Pinocchio (rigid-body dynamics computations)
- Crocoddyl (optimal control library)
- ProxQP (quadratic programming) [OPTIONAL]

# Installation
First clone the repo :

`git clone --recursive https://github.com/machines-in-motion/mim_solvers.git`

  ## Using CMake
`cd mim_solvers && mkdir build && cd build`

`cmake .. [-DCMAKE_BUILD_TYPE=Release] [-DCMAKE_INSTALL_PREFIX=...]`

`make [-j6] && make install`

  ## Using colcon
At the root of your workspace :

`colcon build [optional args]`
