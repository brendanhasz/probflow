"""Tests example correlation"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import probflow as pf

tfd = tfp.distributions


def test_correlation(plot):
    """Tests correlation example"""

    class BayesianCorrelation(pf.Model):
        def __init__(self):
            self.rho = pf.BoundedParameter(min=-1, max=1)

        def __call__(self):
            cov = tf.eye(2) + self.rho() * tf.abs(tf.eye(2) - 1)
            return pf.MultivariateNormal(tf.zeros([2]), cov)

    model = BayesianCorrelation()

    X = np.random.randn(100, 2).astype("float32")
    plt.plot(X[:, 0], X[:, 1], ".")

    if plot:
        plt.show()

    model.fit(X, lr=0.2)
    model.posterior_plot(style="hist", ci=0.95)

    if plot:
        plt.show()

    X[:, 1] = X[:, 0] + 0.2 * np.random.randn(100).astype("float32")
    plt.plot(X[:, 0], X[:, 1], ".")

    if plot:
        plt.show()

    model = BayesianCorrelation()
    model.fit(X, lr=0.2)
    model.posterior_plot(style="hist", ci=0.95)

    if plot:
        plt.show()

    X = np.random.randn(100, 2).astype("float32")
    X[:, 1] = -X[:, 0] + 0.2 * np.random.randn(100).astype("float32")
    plt.plot(X[:, 0], X[:, 1], ".")

    if plot:
        plt.show()

    model = BayesianCorrelation()
    model.fit(X, lr=0.2)
    model.posterior_plot(style="hist", ci=0.95)

    if plot:
        plt.show()
