import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow import applications as apps
from probflow.distributions import Normal
from probflow.models import ContinuousModel, Model

tfd = tfp.distributions


def test_LinearRegression():
    """Tests probflow.applications.LinearRegression"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1

    # Create the model
    model = apps.LinearRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=3)

    # Predictive functions
    preds = model.predict(x[:11, :])
    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 2
    assert preds.shape[0] == 11
    assert preds.shape[1] == 1

    # predictive interval
    lb, ub = model.predictive_interval(x[:12, :], ci=0.9)
    assert isinstance(lb, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 12
    assert lb.shape[1] == 1
    assert isinstance(ub, np.ndarray)
    assert ub.ndim == 2
    assert ub.shape[0] == 12
    assert ub.shape[1] == 1


def test_LinearRegression_heteroscedastic():
    """Tests probflow.applications.LinearRegression w/ heteroscedastic"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1
    y = y + np.exp(y) * np.random.randn(100, 1).astype("float32")

    # Create the model
    model = apps.LinearRegression(5, heteroscedastic=True)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=3)

    # Predictive functions
    preds = model.predict(x[:11, :])
    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 2
    assert preds.shape[0] == 11
    assert preds.shape[1] == 1

    # predictive interval
    lb, ub = model.predictive_interval(x[:12, :], ci=0.9)
    assert isinstance(lb, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 12
    assert lb.shape[1] == 1
    assert isinstance(ub, np.ndarray)
    assert ub.ndim == 2
    assert ub.shape[0] == 12
    assert ub.shape[1] == 1


def test_LinearRegression_multivariate():
    """Tests probflow.applications.LinearRegression w/ d_o>1"""

    # Data
    N = 256
    Di = 7
    Do = 3
    x = np.random.randn(N, Di).astype("float32")
    w = np.random.randn(Di, Do).astype("float32")
    y = x @ w + 0.1 * np.random.randn(N, Do).astype("float32")

    # Create the model
    model = apps.LinearRegression(Di, d_o=Do)

    # Fit the model
    model.fit(x, y, batch_size=128, epochs=3)

    # Predictive functions
    preds = model.predict(x[:11, :])
    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 2
    assert preds.shape[0] == 11
    assert preds.shape[1] == Do

    # Predictive functions
    preds = model.predictive_sample(x[:11, :], n=13)
    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 3
    assert preds.shape[0] == 13
    assert preds.shape[1] == 11
    assert preds.shape[2] == Do
