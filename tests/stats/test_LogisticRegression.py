"""Tests the statistical accuracy of a Logistic Regression w/ ProbFlow"""


import numpy as np

import probflow as pf


def test_logistic_regression():
    """Test that a logistic regression recovers the true parameters"""

    # Set random seed
    # np.random.seed(1234)
    # tf.random.set_seed(1234)

    # Generate data
    N = 1000
    D = 1
    x = np.random.randn(N, D).astype("float32")
    x_val = np.random.randn(N, D).astype("float32")
    w = np.random.randn(D, 1)
    b = np.random.randn()
    noise = 0.1 * np.random.randn(N, 1)
    y = (1.0 / (1.0 + np.exp(-x @ w + b + noise)) > 0.5).astype("float32")
    y_val = (1.0 / (1.0 + np.exp(-x @ w + b + noise)) > 0.5).astype("float32")

    # Create and fit model
    model = pf.LogisticRegression(D)
    model.fit(x, y, batch_size=100, epochs=1000, lr=1e-3)

    # Compute and check confidence intervals on the weights
    lb, ub = model.posterior_ci("weights")
    assert np.all(lb < w)
    assert np.all(ub > w)

    # Compute and check confidence intervals on the bias
    lb, ub = model.posterior_ci("bias")
    assert lb < b
    assert ub > b

    # Compute accuracy (w/ such little noise should be high)
    acc = model.metric("acc", x_val, y_val)
    assert acc > 0.8
