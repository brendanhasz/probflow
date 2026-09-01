"""Tests the statistical accuracy of a Logistic Regression w/ ProbFlow"""

import numpy as np

import probflow as pf

N_DATAPOINTS: int = 1000
N_EPOCHS: int = 1000
BATCH_SIZE: int = 100


def test_logistic_regression():
    """Test that a logistic regression recovers the true parameters."""

    # Set random seed for reproducibility
    np.random.seed(1234)

    # Generate data
    D = 3
    x = np.random.randn(N_DATAPOINTS, D).astype("float32")
    x_val = np.random.randn(N_DATAPOINTS, D).astype("float32")
    w = np.random.randn(D, 1)
    b = np.random.randn()
    # Sample from: y ~ Bernoulli(sigmoid(x @ w + b))
    p = 1.0 / (1.0 + np.exp(-(x @ w + b)))
    y = (np.random.uniform(size=(N_DATAPOINTS, 1)) < p).astype("float32")
    p_val = 1.0 / (1.0 + np.exp(-(x_val @ w + b)))
    y_val = (np.random.uniform(size=(N_DATAPOINTS, 1)) < p_val).astype(
        "float32"
    )

    # Create and fit model
    model = pf.LogisticRegression(D)
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=N_EPOCHS, lr=1e-3)

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
