# mim_solvers
Implementation of numerical solvers used in the Machines in Motion Laboratory. 
In particular, the Sequential Quadratic Programming (SQP) solver described in [this paper](https://laas.hal.science/hal-04330251) solves nonlinear constrained OCPs efficiently.

All solvers are implemented by using [Crocoddyl](https://github.com/loco-3d/crocoddyl/tree/devel) (v2.0) as the base software. 
Consequently, Crocoddyl users can use our efficient solvers while constructing their OCPs using the same API they are used to. 
The default solvers of Crocoddyl are also re-implemented for benchmarking purposes (namely DDP and FDDP) but with modified termination criteria and line-search.

Examples on how to use the solvers can be found in the `examples` directory.

# Dependencies
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) (rigid-body dynamics computations)
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) (optimal control library)
- [ProxSuite](https://github.com/Simple-Robotics/proxsuite) (quadratic programming) [OPTIONAL]

# Installation

  ## Using conda

`conda install mim-solvers --channel conda-forge`


  ## Using CMake
`git clone --recursive https://github.com/machines-in-motion/mim_solvers.git`

`cd mim_solvers && mkdir build && cd build`

`cmake .. [-DCMAKE_BUILD_TYPE=Release] [-DCMAKE_INSTALL_PREFIX=...]`

`make [-j6] && make install`


# Contributors

-   [Armand Jordana](https://github.com/ajordana) (NYU): main developer and manager of the project
-   [Sébastien Kleff](https://github.com/skleff1994) (NYU): main developer and manager of the project
-   [Avadesh Meduri](https://github.com/avadesh02) (NYU): main developer and manager of the project
-   [Ludovic Righetti](https://engineering.nyu.edu/faculty/ludovic-righetti) (NYU): project instructor
-   [Justin Carpentier](https://jcarpent.github.io) (INRIA): project instructor
-   [Nicolas Mansard](http://projects.laas.fr/gepetto/index.php/Members/NicolasMansard) (LAAS-CNRS): project instructor
-   [Yann de Mont-Marin](https://github.com/ymontmarin) (INRIA): Conda integration and support

