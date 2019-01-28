"""Tests a Linear Regression works correctly"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *

N = 1000
epochs = 1000


def test_LR_scalar_no_variance():
    """Tests a LR w/ scalar parameters and no variance parameter"""

    # Model = linear regression assuming error = 1
    weight = Parameter(name='LRsnv_w', estimator=None)
    bias = Parameter(name='LRsnv_b', estimator=None)
    data = Input()
    model = Normal(data*weight + bias, 1.0)

    # Generate data
    true_weight = 0.5
    true_bias = -1
    noise = np.random.randn(N)
    x = np.linspace(-3, 3, N)
    y = true_weight*x + true_bias + noise

    # Fit the model
    model.fit(x, y, epochs=epochs)

    # Ensure we the mean of each parameter individually is correct
    wmean = weight.posterior_mean()
    bmean = bias.posterior_mean()
    assert abs(wmean-true_weight) < 0.2
    assert abs(bmean-true_bias) < 0.2

    # And for the entire model
    means = model.posterior_mean()
    assert isinstance(means, dict)
    assert len(means) == 2
    assert 'LRsnv_w' in means and 'LRsnv_b' in means
    assert abs(means['LRsnv_w']-true_weight) < 0.2
    assert abs(means['LRsnv_b']-true_bias) < 0.2


def test_LR_scalar():
    """Tests a LR w/ scalar parameters"""

    # Model = linear regression assuming error = 1
    weight = Parameter(name='LRs_w', estimator=None)
    bias = Parameter(name='LRs_b', estimator=None)
    std_err = ScaleParameter(name='LRs_s')
    data = Input()
    model = Normal(data*weight + bias, std_err)

    # Generate data
    true_weight = 0.5
    true_bias = -1.0
    true_std_err = 1.0
    noise = np.random.randn(N)
    x = np.linspace(-3, 3, N)
    y = true_weight*x + true_bias + noise*true_std_err

    # Fit the model
    model.fit(x, y, epochs=epochs)

    # Ensure values are correct
    means = model.posterior_mean()
    assert abs(means['LRs_w']-true_weight) < 0.2
    assert abs(means['LRs_b']-true_bias) < 0.2
    assert abs(means['LRs_s']-true_std_err) < 0.2


# TODO: test multivariate LR


# TODO: test multivariate w/ "flipout"


# TODO: test w/ Dense


# TODO: test w/ LinearRegression


#def test_BaseDistribution_sample_posterior_vector():
#    """Tests core.BaseDistribution.sample_posterior w/ vector params"""
#
#    # Parameters + input data is vector of length 3
#    Nd = 3
#
#    # Model = linear regression assuming error = 1
#    weight = Parameter(name='vector_weight', shape=Nd, estimator=None)
#    bias = Parameter(name='nonvec_bias', estimator=None)
#    data = Input()
#    model = Normal(Dot(data, weight) + bias, 1.0)
#
#    # Generate data
#    N = 10
#    true_weight = np.array([0.5, -0.25, 0.0])
#    true_bias = -1.0
#    noise = np.random.randn(N, 1)
#    x = np.random.randn(N, Nd)
#    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise
#
#    # Fit the model
#    model.fit(x, y, epochs=1)
#
#    # Check output of sample_posterior is correct
#    num_samples = 3
#    samples = model.sample_posterior(num_samples=num_samples)
#    assert isinstance(samples, dict)
#    assert len(samples) == 2
#    assert samples['vector_weight'].ndim == 2
#    assert samples['vector_weight'].shape[0] == num_samples
#    assert samples['vector_weight'].shape[1] == Nd
#    assert samples['nonvec_bias'].ndim == 2
#    assert samples['nonvec_bias'].shape[0] == num_samples
#    assert samples['nonvec_bias'].shape[1] == 1

