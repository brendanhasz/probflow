"""Tests probflow.core.BaseDistribution class"""

import pytest

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *

EPOCHS = 2
NUM_SAMPLES = 10


def test_ensure_is_fit(LR1_novar_unfit, LR1_novar):
    """Tests BaseDistribution._ensure_is_fit"""
    with pytest.raises(RuntimeError):
        LR1_novar_unfit._ensure_is_fit()
    LR1_novar._ensure_is_fit()


def test_BaseDistribution_fit(LR1_novar):
    """Tests core.BaseDistribution.fit"""

    model = LR1_novar #fixture from conftest.py

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


def test_BaseDistribution_predictive_distribution(LR3_novar, N_data):
    """Tests core.BaseDistribution.predictive_distribution"""

    model = LR3_novar #fixture from conftest.py

    # Check predictive_distribution with no input
    prd = model.predictive_distribution()
    assert isinstance(prd, np.ndarray)
    assert prd.ndim == 3
    assert prd.shape[0] == 1000
    assert prd.shape[1] == N_data
    assert prd.shape[2] == 1

    # Check predictive_distribution with validation input
    x_val = np.random.rand(4, 3)
    prd = model.predictive_distribution(x_val)
    assert isinstance(prd, np.ndarray)
    assert prd.ndim == 3
    assert prd.shape[0] == 1000
    assert prd.shape[1] == 4
    assert prd.shape[2] == 1

    # Check predictive_distribution w/ val input and num_samples
    prd = model.predictive_distribution(x_val, num_samples=20)
    assert isinstance(prd, np.ndarray)
    assert prd.ndim == 3
    assert prd.shape[0] == 20
    assert prd.shape[1] == 4
    assert prd.shape[2] == 1


def test_BaseDistribution_plot_predictive_distribution(LR3_novar, plot):
    """Tests core.BaseDistribution.plot_predictive_distribution"""

    model = LR3_novar #fixture from conftest.py
    x_val = np.random.rand(10, 3)

    # Check predictive_distribution with no input
    prd = model.plot_predictive_distribution(x_val, style='line')
    if plot:
        plt.suptitle('should show 10 line dists')
        plt.show()

    # Check predictive_distribution with no input
    prd = model.plot_predictive_distribution(x_val, individually=True, cols=2)
    if plot:
        plt.suptitle('should show 5x2 grid of 10 fill dists')
        plt.tight_layout()
        plt.show()

    # Check predictive_distribution with validation input
    x_val = np.random.rand(1, 3)
    prd = model.plot_predictive_distribution(x_val)
    if plot:
        plt.suptitle('should show a single fill dist')
        plt.show()

    # Check predictive_distribution with conf intervals
    prd = model.plot_predictive_distribution(x_val, ci=0.95)
    if plot:
        plt.suptitle('should show a single fill dist w/ 95prc ci')
        plt.show()


def test_BaseDistribution_sample_posterior_scalar(LR1_novar):
    """Tests core.BaseDistribution.sample_posterior w/ scalar params"""

    model = LR1_novar #fixture from conftest.py

    # Check output of sample_posterior is correct
    samples = model.sample_posterior(num_samples=3)
    assert isinstance(samples, dict)
    assert len(samples) == 2
    assert 'LR1_novar_weight' in samples
    assert 'LR1_novar_bias' in samples
    assert type(samples['LR1_novar_weight']) is np.ndarray
    assert samples['LR1_novar_weight'].ndim == 2
    assert samples['LR1_novar_weight'].shape[0] == 3
    assert samples['LR1_novar_weight'].shape[1] == 1
    assert type(samples['LR1_novar_bias']) is np.ndarray
    assert samples['LR1_novar_bias'].ndim == 2
    assert samples['LR1_novar_bias'].shape[0] == 3
    assert samples['LR1_novar_bias'].shape[1] == 1


def test_BaseDistribution_sample_posterior_vector(LR3_novar):
    """Tests core.BaseDistribution.sample_posterior w/ vector params"""

    model = LR3_novar #fixture from conftest.py

    # Check output of sample_posterior is correct
    num_samples = 4
    samples = model.sample_posterior(num_samples=num_samples)
    assert isinstance(samples, dict)
    assert len(samples) == 2
    assert samples['LR3_novar_weight'].ndim == 2
    assert samples['LR3_novar_weight'].shape[0] == num_samples
    assert samples['LR3_novar_weight'].shape[1] == 3
    assert samples['LR3_novar_bias'].ndim == 2
    assert samples['LR3_novar_bias'].shape[0] == num_samples
    assert samples['LR3_novar_bias'].shape[1] == 1


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

    ###  RECORD LIST OF PARAMS, ONCE PER EPOCH  ###

    # Reset
    tf.reset_default_graph()

    # Fit the model, recording all param values once per epoch
    model.fit(x, y, epochs=1, record=['record_weight', 'record_bias'], 
              record_freq='epoch')

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


def test_BaseDistribution_plot_posterior_args_over_training(LR3_var, plot):
    """Tests core.BaseDistribution.plot_posterior_args_over_training"""

    # Parameters + input data is vector of length 3
    Nd = 3

    # Model = linear regression assuming error = 1
    weight = Parameter(name='ppot_weight', shape=Nd, estimator=None)
    bias = Parameter(name='ppot_bias', estimator=None)
    data = Input()
    std_dev = ScaleParameter()
    model = Normal(Dot(data, weight) + bias, std_dev)

    # Generate data
    N = NUM_SAMPLES
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    # All params, once per epoch
    model.fit(x, y, epochs=EPOCHS, record='all', record_freq='epoch')
    model.plot_posterior_args_over_training()
    if plot:
        plt.show()

    # All params, once per batch
    tf.reset_default_graph()
    model.fit(x, y, epochs=20, record='all', record_freq='batch')
    model.plot_posterior_args_over_training(marker='.')
    if plot:
        plt.show()

    # Just weight params
    model.plot_posterior_args_over_training('ppot_weight')
    if plot:
        plt.show()

    # TODO: test w/ 2d params


def test_BaseDistribution_plot_posterior_over_training_scalar(plot):
    """Tests core.BaseDistribution.plot_posterior_over_training 
    w/ scalar params
    """

    # Model = linear regression assuming error = 1
    weight = Parameter(name='ppotp_weight', estimator=None)
    bias = Parameter(name='ppotp_bias', estimator=None)
    data = Input()
    model = Normal(data*weight + bias, 1.0)

    # Generate data
    N = NUM_SAMPLES
    true_weight = 0.5
    true_bias = -1
    noise = np.random.randn(N)
    x = np.linspace(-3, 3, N)
    y = true_weight*x + true_bias + noise

    # Fit
    model.fit(x, y, epochs=EPOCHS, record='all', record_freq='epoch')

    # Plot
    model.plot_posterior_over_training()
    if plot:
        plt.show()


def test_BaseDistribution_plot_posterior_over_training_vector(plot):
    """Tests core.BaseDistribution.plot_posterior_over_training 
    w/ prob=true and vector params
    """

    # Parameters + input data is vector of length 3
    Nd = 3

    # Model = linear regression assuming error = 1
    weight = Parameter(name='ppotpv_weight', shape=Nd, estimator=None)
    bias = Parameter(name='ppotpv_bias', estimator=None)
    data = Input()
    model = Normal(Dot(data, weight) + bias, 1.0)

    # Generate data
    N = NUM_SAMPLES
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    # Fit
    model.fit(x, y, epochs=EPOCHS, record='all', record_freq='epoch')

    # Plot
    model.plot_posterior_over_training()
    if plot:
        plt.show()



# Tests for plot_posterior and plot_prior are in test_plot_posterior/prior


# TODO: test predictive_distribution, predict, metrics, etc
