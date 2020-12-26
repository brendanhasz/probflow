import numpy as np

from probflow.applications import PoissonRegression


def test_PoissonRegression():
    """Tests probflow.applications.PoissonRegression"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1
    y = np.random.poisson(lam=np.exp(y)).astype("float32")

    # Create the model
    model = PoissonRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=3)

    # Predictive functions
    model.predict(x)


def test_PoissonRegression_multiMC_eager():
    """Tests PoissonRegression w/ n_mc>1 in eager mode"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1
    y = np.random.poisson(lam=np.exp(y)).astype("float32")

    # Create the model
    model = PoissonRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=50, epochs=2, n_mc=4, eager=True)

    # Predictive functions
    model.predict(x)


def test_PoissonRegression_multiMC_noneager():
    """Tests PoissonRegression w/ n_mc>1 in non-eager mode"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1
    y = np.random.poisson(lam=np.exp(y)).astype("float32")

    # Create the model
    model = PoissonRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=50, epochs=2, n_mc=4, eager=False)

    # Predictive functions
    model.predict(x)
