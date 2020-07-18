"""Tests the statistical accuracy of a Linear Regression w/ ProbFlow"""


import numpy as np
import tensorflow as tf

import probflow as pf


def test_linear_regression():
    """Test that a linear regression recovers the true parameters"""

    # Set random seed
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # Generate data
    N = 1000
    D = 5
    x = np.random.randn(N, D).astype("float32")
    w = np.random.randn(D, 1)
    b = np.random.randn()
    std = np.exp(np.random.randn())
    noise = std * np.random.randn(N, 1)
    y = x @ w + b + noise
    y = y.astype("float32")

    # Create and fit model
    model = pf.LinearRegression(D)
    model.fit(x, y, batch_size=100, epochs=1000, lr=1e-2)

    # Compute and check confidence intervals on the weights
    lb, ub = model.posterior_ci("weights")
    assert np.all(lb < w)
    assert np.all(ub > w)

    # Compute and check confidence intervals on the bias
    lb, ub = model.posterior_ci("bias")
    assert lb < b
    assert ub > b

    # Compute and check confidence intervals on the std
    lb, ub = model.posterior_ci("std")
    assert lb < std
    assert ub > std
