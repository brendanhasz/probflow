import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from probflow import applications as apps
from probflow.distributions import Normal
from probflow.models import ContinuousModel, Model

tfd = tfp.distributions


def test_PoissonRegression():
    """Tests probflow.applications.PoissonRegression"""

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1
    y = np.random.poisson(lam=np.exp(y)).astype("float32")

    # Create the model
    model = apps.PoissonRegression(5)

    # Fit the model
    model.fit(x, y, batch_size=10, epochs=3)

    # Predictive functions
    model.predict(x)
