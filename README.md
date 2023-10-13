# mim_solvers
Implementation of numerical solvers used in the Machines in Motion Laboratory. In particular, the Sequential Quadratic Programming (SQP) solver decribed in [arxiv] that solves efficiently nonlinear constrained OCPs.
Solvers from Crocoddyl are also available, namely DDP and FDDP, but with modified termination criteria and line-search.

# Dependencies
- Pinocchio (rigid-body dynamics computations)
- Crocoddyl (optimal control library)
- ProxQP (quadratic programming) [OPTIONAL]

# Installation
First clone the repo :

`git clone https://github.com/machines-in-motion/mim_solvers.git`

  ## Using CMake
`cd mim_solvers && mkdir build && cd build`

`cmake .. [-DCMAKE_BUILD_TYPE=Release] [-DCMAKE_INSTALL_PREFIX=...]`

`make [-j6] && make install`

  ## Using colcon
At the root of your workspace :

`colcon build [optional args]`
