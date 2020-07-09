"""Tests example robust heteroscedastic regression"""


import pytest

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import probflow as pf

tfd = tfp.distributions


def test_correlation(plot):
    """Tests correlation example"""

    class RobustHeteroscedasticRegression(pf.ContinuousModel):
        def __init__(self, dims):
            self.w = pf.Parameter([dims, 2])
            self.b = pf.Parameter(2)

        def __call__(self, x):
            p = x @ self.w() + self.b()
            means = p[:, 0]
            stds = tf.exp(p[:, 1])
            return pf.Cauchy(means, stds)

    model = RobustHeteroscedasticRegression(3)

    X = np.random.randn(100, 3)
    y = np.random.randn(100, 1)

    model.fit(X, y)

    # TODO: check output looks correct
    # if plot:
    #    plt.show()
