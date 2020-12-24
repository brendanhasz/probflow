"""Tests exmaple gaussian mixture model"""


import numpy as np
import tensorflow_probability as tfp

import probflow as pf

tfd = tfp.distributions


def is_close(a, b, tol=1e-3):
    return np.abs(a - b) < tol


def test_gmm():
    """Tests gaussian mixture model example"""

    class GaussianMixtureModel(pf.Model):
        def __init__(self, d, k):
            self.m = pf.Parameter([d, k])
            self.s = pf.ScaleParameter([d, k])
            self.w = pf.DirichletParameter(k)

        def __call__(self):
            return pf.Mixture(pf.Normal(self.m(), self.s()), probs=self.w())

    model = GaussianMixtureModel(2, 3)

    # Create some data
    X = np.concatenate(
        [
            np.random.randn(100, 2),
            np.random.randn(100, 2) + 5,
            np.random.randn(100, 2) + 10,
        ],
        axis=0,
    )
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Fit the model
    model.fit(X)

    # TODO: check it has fit correctly
    # import pdb; pdb.set_trace()
