"""
License: BSD 3-Clause License

This file checks the bounds and the initial value of CSQP's adaptive rho.
"""

import crocoddyl
import mim_solvers
import numpy as np

model = crocoddyl.ActionModelLQR(2, 1)
problem = crocoddyl.ShootingProblem(np.zeros(2), [model] * 3, crocoddyl.ActionModelLQR(2, 0))
solver = mim_solvers.SolverCSQP(problem)

assert (solver.rho_min, solver.rho_max, solver.rho_sparse_base) == (1e-6, 1e3, 1e-1)
solver.rho_max = 1e4
solver.rho_min = 100.0
solver.rho_sparse_base = 100.0
assert (solver.rho_min, solver.rho_max, solver.rho_sparse_base) == (100.0, 1e4, 100.0)

for name, value in (("rho_min", 0.0), ("rho_min", 1e5), ("rho_max", 1.0),
                    ("rho_sparse_base", 1.0), ("rho_sparse_base", 1e5)):
    try:
        setattr(solver, name, value)
    except Exception:
        pass
    else:
        raise AssertionError(f"{name} = {value} should be rejected")

# Each solve starts from rho_sparse_base, the adaptive rho stays within the bounds.
solver.solve([np.ones(2)] * 4, [np.ones(1)] * 3, 3, False)
assert solver.rho_min <= solver.rho_sparse <= solver.rho_max


# reset_direction() called from a callback rejects the step: the solve stops.
class RejectSteps(mim_solvers.CallbackAbstract):
    def __init__(self):
        mim_solvers.CallbackAbstract.__init__(self)

    def __call__(self, solver, solver_type=None):
        solver.reset_direction()
        assert not any(np.any(dx) for dx in solver.dx)


rejecting = mim_solvers.SolverCSQP(problem)
rejecting.setCallbacks([RejectSteps()])
x_init = [np.ones(2)] * 4
rejecting.solve(x_init, [np.ones(1)] * 3, 5, False)
assert rejecting.iter == 0
xs = [np.array(x) for x in rejecting.xs]  # copies: the vector is a temporary
assert all(np.allclose(x, np.ones(2)) for x in xs[1:])
