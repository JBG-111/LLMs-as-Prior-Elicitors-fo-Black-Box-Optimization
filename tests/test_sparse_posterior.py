import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class SparsePosteriorTests(unittest.TestCase):
    def test_fit_is_deterministic_and_learns_preference_direction(self):
        from sparse_preference_bo import SparsePreferencePosterior

        x = np.linspace(-2.0, 2.0, 9)[:, None]
        pairs = np.asarray([[i + 1, i] for i in range(8)], dtype=int)
        labels = np.ones(8)
        kwargs = dict(n_inducing=7, epochs=30, batch_size=4, learning_rate=0.03, seed=4)
        first = SparsePreferencePosterior.fit(x, pairs, labels, **kwargs)
        second = SparsePreferencePosterior.fit(x, pairs, labels, **kwargs)
        mean, variance = first.mean_and_variance(x)

        np.testing.assert_allclose(first.mean_weights, second.mean_weights)
        self.assertGreater(mean[-1], mean[0])
        self.assertTrue(np.all(variance > 0.0))

    def test_covariance_is_symmetric(self):
        from sparse_preference_bo import SparsePreferencePosterior

        x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        posterior = SparsePreferencePosterior.fit(
            x,
            np.asarray([[1, 0], [2, 1], [3, 2]]),
            np.ones(3),
            n_inducing=4,
            epochs=4,
            batch_size=3,
            seed=2,
        )
        covariance = posterior.covariance(x, x)
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
        self.assertTrue(np.all(np.diag(covariance) > 0.0))


if __name__ == "__main__":
    unittest.main()
