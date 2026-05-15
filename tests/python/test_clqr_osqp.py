"""
License: BSD 3-Clause License
Copyright (C) 2024, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.

This file checks that all the custom OSQP implementation matches the official one.
"""

import numpy as np
import unittest

from csqp import CSQP  # noqa: E402
from problems import create_clqr_problem  # noqa: E402

LINE_WIDTH = 100

print(" TEST OSQP ".center(LINE_WIDTH, "-"))


class TestCLQROSQP(unittest.TestCase):
    def setUp(self):
        self.problem, self.xs_init, self.us_init = create_clqr_problem()
        self.ddp1 = CSQP(self.problem, "CustomOSQP")
        self.ddp2 = CSQP(self.problem, "OSQP")
        self.ddp1.with_callbacks = True
        self.ddp2.with_callbacks = True
        max_qp_iters = 25
        self.ddp1.max_qp_iters = max_qp_iters
        self.ddp2.max_qp_iters = max_qp_iters
        eps_abs = 1e-8
        self.ddp1.eps_abs = eps_abs
        self.ddp2.eps_abs = eps_abs

    def test_osqp_match(self):
        self.ddp1.solve(self.xs_init, self.us_init, 1)
        self.ddp2.solve(self.xs_init, self.us_init, 1)
        set_tol = 1e-8
        self.assertEqual(self.ddp1.qp_iters, self.ddp2.qp_iters)
        self.assertLess(
            np.linalg.norm(np.array(self.ddp1.xs) - np.array(self.ddp2.xs)),
            set_tol,
            "Test failed: xs mismatch",
        )
        self.assertLess(
            np.linalg.norm(np.array(self.ddp1.us) - np.array(self.ddp2.us)),
            set_tol,
            "Test failed: us mismatch",
        )
        self.assertLess(
            np.linalg.norm(np.array(self.ddp1.lag_mul) - np.array(self.ddp2.lag_mul)),
            set_tol,
            "Test failed: lag_mul mismatch",
        )
        for t in range(len(self.ddp1.y)):
            self.assertLess(
                np.linalg.norm(self.ddp1.y[t] - self.ddp2.y[t]),
                set_tol,
                f"Test failed: y mismatch at t={t}",
            )


if __name__ == "__main__":
    unittest.main()
