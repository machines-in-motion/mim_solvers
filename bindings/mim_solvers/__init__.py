"""__init__
License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

from .mim_solvers_pywrap import *  # noqa
import copy
import numpy as np


class CallbackLogger(CallbackAbstract):  # noqa: F405
    def __init__(self, verbose=False):
        super().__init__()
        self.convergence_data = {}

    def __call__(self, solver, solver_type):
        def safe_append(key, value):
            if key not in self.convergence_data:
                self.convergence_data[key] = []
            self.convergence_data[key].append(value)

        if solver_type == "CSQP":
            safe_append("us", copy.copy((solver.us)))
            safe_append("xs", copy.copy((solver.xs)))
            safe_append("fs", copy.copy((solver.fs)))
            safe_append("iter", solver.iter)
            safe_append("cost", solver.cost)
            safe_append("merit", solver.merit)
            safe_append("preg", solver.preg)
            safe_append("dreg", solver.dreg)
            safe_append("step", solver.stepLength)
            safe_append("gap_norm", solver.gap_norm)
            safe_append("constraint_norm", solver.constraint_norm)
            safe_append("qp_iter", solver.qp_iters)
            safe_append("KKT", solver.KKT)
        elif solver_type == "SQP":
            safe_append("us", copy.copy((solver.us)))
            safe_append("xs", copy.copy((solver.xs)))
            safe_append("fs", copy.copy((solver.fs)))
            safe_append("iter", solver.iter)
            safe_append("cost", solver.cost)
            safe_append("merit", solver.merit)
            safe_append("preg", solver.preg)
            safe_append("dreg", solver.dreg)
            safe_append("step", solver.stepLength)
            safe_append("KKT", solver.KKT)
        else:
            raise NotImplementedError("CallbackLogger is implemented for CSQP and SQP.")


def plotConvergence(data, show=True):
    import matplotlib.pyplot as plt

    axis = 0
    for i, (key, values) in enumerate(data.items()):
        if len(np.asarray(values, dtype="object").shape) == 1:
            axis += 1
    fig, axs = plt.subplots(
        axis - 2, 1, sharex="col", figsize=(55, 25.5)
    )  # exclude dreg and preg

    i = 0
    for key, values in data.items():
        if len(np.asarray(values, dtype="object").shape) == 1:
            if key == "dreg" or key == "preg":
                continue

            axs[i].plot(values)
            axs[i].set_title(key)
            if key == "qp_iter":
                axs[i].text(
                    0.9,
                    0.9,
                    f"Total qp_iters: {sum(values)}",
                    transform=axs[i].transAxes,
                    ha="right",
                )
                axs[i].text(
                    0.9,
                    0.8,
                    f"Total sqp_iters: {len(values)}",
                    transform=axs[i].transAxes,
                    ha="right",
                )

            if key == "KKT":
                axs[i].set_yscale("log")
            i += 1
    plt.tight_layout()

    if show:
        plt.show()

    return fig
