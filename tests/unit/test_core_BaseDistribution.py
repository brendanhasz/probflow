"""Tests probflow.core.BaseDistribution class"""

import pytest

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *

PLOT = False
EPOCHS = 1
NUM_SAMPLES = 10


def test_BaseDistribution_fit():
    """Tests core.BaseDistribution.fit"""

    # Model = linear regression assuming error = 1
    weight = Parameter(estimator=None)
    bias = Parameter(estimator=None)
    data = Input()
    model = Normal(data*weight + bias, 1.0)

    # Generate data
    N = 10
    true_weight = 0.5
    true_bias = -1
    noise = np.random.randn(N)
    x = np.linspace(-3, 3, N)
    y = true_weight*x + true_bias + noise

    # Should return a RuntimeError
    with pytest.raises(RuntimeError):
        model._ensure_is_fit()

    # Fit the model
    model.fit(x, y, epochs=1)

    # Now should pass
    model._ensure_is_fit()

    # Check model contains data
    assert isinstance(model._train, dict)
    assert isinstance(model._train['x'], np.ndarray)
    assert isinstance(model._train['y'], np.ndarray)
    assert type(model._shuffled_ids) is np.ndarray
    assert isinstance(model._ph, dict)
    assert isinstance(model._ph['batch_size'], tf.Tensor)
    assert isinstance(model._ph['x'], tf.Tensor)
    assert isinstance(model._ph['y'], tf.Tensor)
    assert type(model.log_loss) is tf.Tensor
    assert type(model.mean_log_loss) is tf.Tensor
    assert type(model.kl_loss) is tf.Tensor
    assert type(model._session) is tf.Session
    assert model.is_fit


def test_BaseDistribution_sample_posterior_scalar():
    """Tests core.BaseDistribution.sample_posterior w/ scalar params"""

    # Model = linear regression assuming error = 1
    weight = Parameter(name='weight', estimator=None)
    bias = Parameter(name='bias', estimator=None)
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

    # Check output of sample_posterior is correct
    samples = model.sample_posterior(num_samples=3)
    assert isinstance(samples, dict)
    assert len(samples) == 2
    assert 'weight' in samples and 'bias' in samples
    assert type(samples['weight']) is np.ndarray
    assert samples['weight'].ndim == 2
    assert samples['weight'].shape[0] == 3
    assert samples['weight'].shape[1] == 1
    assert type(samples['bias']) is np.ndarray
    assert samples['bias'].ndim == 2
    assert samples['bias'].shape[0] == 3
    assert samples['bias'].shape[1] == 1


def test_BaseDistribution_sample_posterior_vector():
    """Tests core.BaseDistribution.sample_posterior w/ vector params"""

    # Parameters + input data is vector of length 3
    Nd = 3

    # Model = linear regression assuming error = 1
    weight = Parameter(name='vector_weight', shape=Nd, estimator=None)
    bias = Parameter(name='nonvec_bias', estimator=None)
    data = Input()
    model = Normal(Dot(data, weight) + bias, 1.0)

    # Generate data
    N = 10
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    # Fit the model
    model.fit(x, y, epochs=1)

    # Check output of sample_posterior is correct
    num_samples = 3
    samples = model.sample_posterior(num_samples=num_samples)
    assert isinstance(samples, dict)
    assert len(samples) == 2
    assert samples['vector_weight'].ndim == 2
    assert samples['vector_weight'].shape[0] == num_samples
    assert samples['vector_weight'].shape[1] == Nd
    assert samples['nonvec_bias'].ndim == 2
    assert samples['nonvec_bias'].shape[0] == num_samples
    assert samples['nonvec_bias'].shape[1] == 1


def test_BaseDistribution_sample_posterior_vector_pandas():
    """Tests core.BaseDistribution.sample_posterior + fit w/ pandas input"""

    # Parameters + input data is vector of length 3
    Nd = 3

    # Model = linear regression assuming error = 1
    weight = Parameter(name='pd_weight', shape=Nd, estimator=None)
    bias = Parameter(name='pd_bias', estimator=None)
    data = Input()
    model = Normal(Dot(data, weight) + bias, 1.0)

    # Generate data
    N = 10
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N)
    x = np.random.randn(N, Nd+1)
    x[:,0] = np.sum(true_weight*x[:,1:], axis=1) + true_bias + noise
    df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])

    # Fit the model
    model.fit(['b', 'c', 'd'], 'a', data=df, epochs=1)

    # Check output of sample_posterior is correct
    num_samples = 3
    samples = model.sample_posterior(num_samples=num_samples)
    assert isinstance(samples, dict)
    assert len(samples) == 2
    assert samples['pd_weight'].ndim == 2
    assert samples['pd_weight'].shape[0] == num_samples
    assert samples['pd_weight'].shape[1] == Nd
    assert samples['pd_bias'].ndim == 2
    assert samples['pd_bias'].shape[0] == num_samples
    assert samples['pd_bias'].shape[1] == 1



# TODO: test 2D X and params


# TODO: test vector Y


# TODO: test 2D Y


# TODO: test the shuffles work _initialize_shuffles


# TODO: test that the flipout estimator works?


# TODO: test the batches are being generated correctly _generate_batch



def test_BaseDistribution_fit_record():
    """Tests core.BaseDistribution.fit w/ record-related args"""

    # Parameters + input data is vector of length 3
    Nd = 3

    # Model = linear regression assuming error = 1
    weight = Parameter(name='record_weight', shape=Nd, estimator=None)
    bias = Parameter(name='record_bias', estimator=None)
    data = Input()
    model = Normal(Dot(data, weight) + bias, 1.0)

    # Generate data
    N = 10
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    ###  NO RECORDING  ###

    # Fit the model
    model.fit(x, y, epochs=1, record=None)

    # No recording should have occurred
    assert not hasattr(model, '_records')

    ###  RECORD ALL, ONCE PER EPOCH  ###

    # Reset
    tf.reset_default_graph()

    # Fit the model, recording all param values once per epoch
    model.fit(x, y, epochs=1, batch_size=5, record='all', record_freq='epoch')

    # Check records
    assert hasattr(model, '_records')
    assert isinstance(model._records, dict)
    assert 'record_weight' in model._records
    assert isinstance(model._records['record_weight'], dict)
    assert 'loc' in model._records['record_weight']
    assert 'scale' in model._records['record_weight']
    assert isinstance(model._records['record_weight']['loc'], np.ndarray)
    assert isinstance(model._records['record_weight']['scale'], np.ndarray)
    assert model._records['record_weight']['loc'].ndim == 2
    assert model._records['record_weight']['loc'].shape[0] == 1
    assert model._records['record_weight']['loc'].shape[1] == 3
    assert model._records['record_weight']['scale'].ndim == 2
    assert model._records['record_weight']['scale'].shape[0] == 1
    assert model._records['record_weight']['scale'].shape[1] == 3
    assert 'record_bias' in model._records
    assert isinstance(model._records['record_bias'], dict)
    assert 'loc' in model._records['record_bias']
    assert 'scale' in model._records['record_bias']
    assert isinstance(model._records['record_bias']['loc'], np.ndarray)
    assert isinstance(model._records['record_bias']['scale'], np.ndarray)
    assert model._records['record_bias']['loc'].ndim == 2
    assert model._records['record_bias']['loc'].shape[0] == 1
    assert model._records['record_bias']['loc'].shape[1] == 1
    assert model._records['record_bias']['scale'].ndim == 2
    assert model._records['record_bias']['scale'].shape[0] == 1
    assert model._records['record_bias']['scale'].shape[1] == 1

    ###  RECORD ALL, ONCE PER BATCH  ###

    # Reset
    tf.reset_default_graph()

    # Fit the model, recording all param values once per epoch
    model.fit(x, y, epochs=2, batch_size=5, record='all', record_freq='batch')

    # Check records
    assert hasattr(model, '_records')
    assert isinstance(model._records, dict)
    assert 'record_weight' in model._records
    assert isinstance(model._records['record_weight'], dict)
    assert 'loc' in model._records['record_weight']
    assert 'scale' in model._records['record_weight']
    assert isinstance(model._records['record_weight']['loc'], np.ndarray)
    assert isinstance(model._records['record_weight']['scale'], np.ndarray)
    assert model._records['record_weight']['loc'].ndim == 2
    assert model._records['record_weight']['loc'].shape[0] == 4
    assert model._records['record_weight']['loc'].shape[1] == 3
    assert model._records['record_weight']['scale'].ndim == 2
    assert model._records['record_weight']['scale'].shape[0] == 4
    assert model._records['record_weight']['scale'].shape[1] == 3
    assert 'record_bias' in model._records
    assert isinstance(model._records['record_bias'], dict)
    assert 'loc' in model._records['record_bias']
    assert 'scale' in model._records['record_bias']
    assert isinstance(model._records['record_bias']['loc'], np.ndarray)
    assert isinstance(model._records['record_bias']['scale'], np.ndarray)
    assert model._records['record_bias']['loc'].ndim == 2
    assert model._records['record_bias']['loc'].shape[0] == 4
    assert model._records['record_bias']['loc'].shape[1] == 1
    assert model._records['record_bias']['scale'].ndim == 2
    assert model._records['record_bias']['scale'].shape[0] == 4
    assert model._records['record_bias']['scale'].shape[1] == 1

    ###  RECORD JUST ONE PARAM, ONCE PER EPOCH  ###

    # Reset
    tf.reset_default_graph()

    # Fit the model, recording all param values once per epoch
    model.fit(x, y, epochs=1, record='record_weight', record_freq='epoch')

    # Check records
    assert hasattr(model, '_records')
    assert isinstance(model._records, dict)
    assert 'record_weight' in model._records
    assert isinstance(model._records['record_weight'], dict)
    assert 'loc' in model._records['record_weight']
    assert 'scale' in model._records['record_weight']
    assert isinstance(model._records['record_weight']['loc'], np.ndarray)
    assert isinstance(model._records['record_weight']['scale'], np.ndarray)
    assert model._records['record_weight']['loc'].ndim == 2
    assert model._records['record_weight']['loc'].shape[0] == 1
    assert model._records['record_weight']['loc'].shape[1] == 3
    assert model._records['record_weight']['scale'].ndim == 2
    assert model._records['record_weight']['scale'].shape[0] == 1
    assert model._records['record_weight']['scale'].shape[1] == 3
    assert 'record_bias' not in model._records


def test_BaseDistribution_plot_posterior_over_training():
    """Tests core.BaseDistribution.plot_posterior_over_training"""

    # Parameters + input data is vector of length 3
    Nd = 3

    # Model = linear regression assuming error = 1
    weight = Parameter(name='record_weight', shape=Nd, estimator=None)
    bias = Parameter(name='record_bias', estimator=None)
    data = Input()
    model = Normal(Dot(data, weight) + bias, 1.0)

    # Generate data
    N = NUM_SAMPLES
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    ###  RECORD ALL, ONCE PER EPOCH  ###

    # Fit the model, recording all param values once per epoch
    model.fit(x, y, epochs=EPOCHS, record='all', record_freq='epoch')

    # Plot records
    model.plot_posterior_over_training(prob=False)
    plt.show()


    # TODO: test w/ once per batch


    # TODO: test w/ only certain args


    # TODO: test when prob=True)


    # Reset
    tf.reset_default_graph()

    # Fit the model, recording all param values once per epoch
    #model.fit(x, y, epochs=2, batch_size=5, record='all', record_freq='batch')

    # Plot records
    #model.plot_posterior_over_training()
    #plt.show()


# Tests for plot_posterior and plot_prior are in test_plot_posterior/prior


# TODO: test predictive_distribution, predict, metrics, etc


if __name__ == "__main__":
    PLOT = True
    EPOCHS = 1000
    NUM_SAMPLES = 1000
    import matplotlib.pyplot as plt
    test_BaseDistribution_plot_posterior_over_training()
