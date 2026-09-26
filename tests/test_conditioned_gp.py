import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ConditionedGPTests(unittest.TestCase):
    def test_no_observations_returns_prior_marginals(self):
        from sparse_preference_bo.conditioned_gp import RBFZeroPrior, conditioned_moments

        support = np.asarray([[0.0], [1.0], [2.0]])
        prior = RBFZeroPrior.fit(support, seed=0)
        query = np.asarray([[0.25], [1.75]])
        mean, variance = conditioned_moments(
            prior, query, np.empty((0, 1)), np.empty(0)
        )
        np.testing.assert_allclose(mean, prior.mean(query))
        np.testing.assert_allclose(variance, prior.variance(query))

    def test_observations_condition_the_latent_utility(self):
        from sparse_preference_bo.conditioned_gp import RBFZeroPrior, conditioned_moments

        support = np.linspace(0.0, 1.0, 8)[:, None]
        prior = RBFZeroPrior.fit(support, seed=0)
        query = np.asarray([[0.0], [1.0]])
        mean, variance = conditioned_moments(
            prior,
            query,
            np.asarray([[0.0], [1.0]]),
            np.asarray([0.0, 2.0]),
        )
        self.assertGreater(mean[0], mean[1])
        self.assertTrue(np.all(variance > 0.0))

    def test_acquisition_scores_are_finite_and_ei_is_nonnegative(self):
        from sparse_preference_bo.acquisition import acquisition_scores

        mean = np.asarray([-0.5, 0.0, 0.75])
        variance = np.asarray([0.2, 0.3, 0.4])
        observed = np.asarray([2.0, 1.0, 1.5])
        ei = acquisition_scores("ei", mean, variance, observed)
        confidence_bound = acquisition_scores("lcb", mean, variance, observed)
        self.assertTrue(np.all(np.isfinite(ei)))
        self.assertTrue(np.all(ei >= 0.0))
        self.assertTrue(np.all(np.isfinite(confidence_bound)))


if __name__ == "__main__":
    unittest.main()
