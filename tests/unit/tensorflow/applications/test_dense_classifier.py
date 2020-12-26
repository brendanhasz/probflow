import numpy as np

from probflow.applications import DenseClassifier


def get_multinomial_data(n, di, do):
    x = np.random.randn(n, di).astype("float32")
    w = np.random.randn(di, do).astype("float32")
    b = np.random.randn(1, do).astype("float32")
    y = np.argmax(x @ w + b, axis=1).astype("int32")
    return x, y


def test_DenseClassifier():
    """Tests probflow.applications.DenseClassifier"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1
    y = np.round(1.0 / (1.0 + np.exp(-y))).astype("float32")

    # Create the model
    model = DenseClassifier([5, 20, 15, 2])

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=3)

    # Predictive functions
    model.predict(x)


def test_MultinomialDenseClassifier():
    """Tests probflow.applications.DenseClassifier w/ >2 output classes"""

    # Data
    x, y = get_multinomial_data(100, 5, 3)

    # Create the model
    model = DenseClassifier([5, 10, 7, 3])

    # Fit the model
    model.fit(x, y, batch_size=25, epochs=3)

    # Predictive functions
    model.predict(x)


def test_DenseClassifier_multimc_eager():
    """Tests DenseClassifier w/ n_mc>1 in eager mode"""

    # Data
    x, y = get_multinomial_data(100, 5, 3)

    # Create the model
    model = DenseClassifier([5, 10, 7, 3])

    # Fit the model
    model.fit(x, y, batch_size=50, epochs=2, n_mc=4, eager=True)

    # Predictive functions
    model.predict(x)


def test_DenseClassifier_multimc_noneager():
    """Tests DenseClassifier w/ n_mc>1 in non-eager mode"""

    # Data
    x, y = get_multinomial_data(100, 5, 3)

    # Create the model
    model = DenseClassifier([5, 10, 7, 3])

    # Fit the model
    model.fit(x, y, batch_size=50, epochs=2, n_mc=4, eager=False)

    # Predictive functions
    model.predict(x)
