# mim_solvers
Implementation of numerical solvers used in the Machines in Motion Laboratory. In particular, the Sequential Quadratic Programming (SQP) solver decribed in [arxiv] that solves efficiently nonlinear constrained OCPs.

NOTE: currently, only the un-constrained solver is available (a.k.a. GNMS) . The constrained version will be added to this repo once it has been updated to Crocoddyl v2 (ongoing work)

# Dependencies
- Pinocchio (rigid-body dynamics computations)
- Crocoddyl (optimal control library)
- ProxQP (quadratic programming) [OPTIONAL]
