"""Tests probflow.core.BaseDistribution class"""

import pytest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *


def test_BaseDistribution_fit():
    """Tests core.BaseDistribution.fit"""

    # Model = linear regression assuming error = 1
    weight = Parameter()
    bias = Parameter()
    data = Input()
    model = Normal(data*weight + bias, 1.0)

    # Generate data
    N = 10
    true_weight = 0.5
    true_bias = -1
    noise = np.random.randn(N)
    x = np.linspace(-3, 3, N)
    y = true_weight*x + true_bias + noise

    # Fit the model
    model.fit(x, y, epochs=1)



def test_BaseDistribution_sample_posterior():
    """Tests core.BaseDistribution.sample_posterior"""

    # Model = linear regression assuming error = 1
    weight = Parameter(name='weight', shape=2)
    bias = Parameter(name='bias', shape=2)
    data = Input()
    model = Normal(data*weight + bias, 1.0)

    # Generate data
    N = 10
    true_weight = np.array([0.5, -0.25])
    true_bias = np.array([-1, 0.75])
    noise = np.random.randn(N,2)
    x = np.random.randn(N,2)
    y = true_weight*x + true_bias + noise

    # Fit the model
    model.fit(x, y, epochs=1)

    # Check output of sample_posterior is correct
    samples = model.sample_posterior(num_samples=3)
    assert isinstance(samples, dict)
    assert len(samples) == 2
    assert 'weight' in samples and 'bias' in samples
    assert isinstance(samples['weight'], np.ndarray)
    assert samples['weight'].ndim == 2
    assert samples['weight'].shape[0] == 3
    assert samples['weight'].shape[1] == 2
    assert isinstance(samples['bias'], np.ndarray)
    assert samples['bias'].ndim == 2
    assert samples['bias'].shape[0] == 3
    assert samples['bias'].shape[1] == 2
