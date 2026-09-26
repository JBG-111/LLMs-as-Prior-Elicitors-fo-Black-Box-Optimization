import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class BODriverTests(unittest.TestCase):
    def test_finite_pool_never_reselects_candidates(self):
        from sparse_preference_bo.conditioned_gp import RBFZeroPrior
        from sparse_preference_bo.finite_pool_bo import run_finite_pool_bo

        x = np.linspace(-2.0, 2.0, 21)[:, None]
        objective = (x[:, 0] - 0.4) ** 2
        prior = RBFZeroPrior.fit(x, seed=3)
        result = run_finite_pool_bo(
            prior,
            x,
            objective,
            initial_indices=[0, 20],
            n_steps=5,
            acquisition="ei",
            seed=7,
        )
        self.assertEqual(result["selected_indices"][:2], [0, 20])
        self.assertEqual(len(result["selected_indices"]), 7)
        self.assertEqual(len(set(result["selected_indices"])), 7)
        self.assertEqual(len(result["best_objective"]), 5)

    def test_continuous_proposal_is_bounded_and_deterministic(self):
        from sparse_preference_bo.conditioned_gp import RBFZeroPrior
        from sparse_preference_bo.continuous_bo import propose

        support = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
        prior = RBFZeroPrior.fit(support, seed=2)
        bounds = np.asarray([[0.0, -1.0], [1.0, 2.0]])
        train_x = np.asarray([[0.2, 0.0], [0.8, 1.0]])
        train_y = np.asarray([1.0, 0.2])
        first = propose("lcb", prior, train_x, train_y, bounds, seed=10, n_raw=128)
        second = propose("lcb", prior, train_x, train_y, bounds, seed=10, n_raw=128)
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(first >= bounds[0]))
        self.assertTrue(np.all(first <= bounds[1]))


if __name__ == "__main__":
    unittest.main()
