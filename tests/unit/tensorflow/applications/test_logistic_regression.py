import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow import applications as apps
from probflow.distributions import Normal
from probflow.models import ContinuousModel, Model

tfd = tfp.distributions


def test_LogisticRegression():
    """Tests probflow.applications.LinearRegression"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1
    y = np.round(1.0 / (1.0 + np.exp(-y))).astype("int32")

    # Create the model
    model = apps.LogisticRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=3)

    # Predictive functions
    model.predict(x)


def test_MultinomialLogisticRegression():
    """Tests probflow.applications.LinearRegression w/ >2 output classes"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 3).astype("float32")
    b = np.random.randn(1, 3).astype("float32")
    y = x @ w + b
    y = np.argmax(y, axis=1).astype("int32")

    # Create the model
    model = apps.LogisticRegression(d=5, k=3)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=3)

    # Predictive functions
    model.predict(x)
