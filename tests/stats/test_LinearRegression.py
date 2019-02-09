"""Tests a Linear Regression works correctly"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *

PLOT = False
N = 1000
epochs = 1000


def test_LR_scalar_no_variance():
    """Tests a LR w/ scalar parameters and no variance parameter"""

    # Model = linear regression assuming error = 1
    weight = Parameter(name='LRsnv_weight', estimator=None)
    bias = Parameter(name='LRsnv_bias', estimator=None)
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
    assert 'LRsnv_weight' in means and 'LRsnv_bias' in means
    assert abs(means['LRsnv_weight']-true_weight) < 0.2
    assert abs(means['LRsnv_bias']-true_bias) < 0.2

    # Plot
    if PLOT:
        model.plot_posterior(ci=0.95)
        plt.suptitle('Linear Regression - ' + 
                     'weight=' + str(true_weight) +
                     ', bias=' + str(true_bias))
        plt.show()


def test_LR_scalar():
    """Tests a LR w/ scalar parameters"""

    # Model = linear regression assuming error = 1
    weight = Parameter(name='LRs_weight', estimator=None)
    bias = Parameter(name='LRs_bias', estimator=None)
    std_err = ScaleParameter(name='LRs_std_dev')
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
    assert abs(means['LRs_weight']-true_weight) < 0.2
    assert abs(means['LRs_bias']-true_bias) < 0.2
    assert abs(means['LRs_std_dev']-true_std_err) < 0.2

    # Plot
    if PLOT:
        model.plot_posterior(ci=0.95)
        plt.suptitle('Linear Regression - ' + 
                     'weight=' + str(true_weight) +
                     ', bias=' + str(true_bias) +
                     ', std_err=' + str(true_std_err))
        plt.show()


def test_LR_vector():
    """Tests a LR w/ vector weight parameter/input"""

    Nd = 3 #number of dimensions of input

    # Model = linear regression assuming error = 1
    weight = Parameter(shape=Nd, name='LRv_weight', estimator=None)
    bias = Parameter(name='LRv_bias', estimator=None)
    std_err = ScaleParameter(name='LRv_std_dev')
    data = Input()
    model = Normal(Dot(data, weight) + bias, std_err)

    # Generate data
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    true_std_err = 1.0
    noise = true_std_err*np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    # Fit the model
    model.fit(x, y, epochs=epochs)

    # Ensure values are correct
    means = model.posterior_mean()
    assert abs(means['LRv_weight'][0]-true_weight[0]) < 0.2
    assert abs(means['LRv_weight'][1]-true_weight[1]) < 0.2
    assert abs(means['LRv_weight'][2]-true_weight[2]) < 0.2
    assert abs(means['LRv_bias']-true_bias) < 0.2
    assert abs(means['LRv_std_dev']-true_std_err) < 0.2

    # Plot
    if PLOT:
        model.plot_posterior(ci=0.95)
        plt.suptitle('Linear Regression - ' + 
                     'weights=' + str(true_weight) +
                     ', bias=' + str(true_bias) +
                     ', std_err=' + str(true_std_err))
        plt.show()


def test_LR_vector_flipout():
    """Tests a LR w/ vector weight parameter/input + 'flipout' estimator"""

    Nd = 3 #number of dimensions of input

    # Model = linear regression assuming error = 1
    weight = Parameter(shape=Nd, name='LRvf_weight')
    bias = Parameter(name='LRvf_bias')
    std_err = ScaleParameter(name='LRvf_std_dev')
    data = Input()
    model = Normal(Dot(data, weight) + bias, std_err)

    # Generate data
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    true_std_err = 1.0
    noise = true_std_err*np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    # Fit the model
    model.fit(x, y, epochs=epochs)

    # Ensure values are correct
    means = model.posterior_mean()
    assert abs(means['LRvf_weight'][0]-true_weight[0]) < 0.2
    assert abs(means['LRvf_weight'][1]-true_weight[1]) < 0.2
    assert abs(means['LRvf_weight'][2]-true_weight[2]) < 0.2
    assert abs(means['LRvf_bias']-true_bias) < 0.2
    assert abs(means['LRvf_std_dev']-true_std_err) < 0.2

    # Plot
    if PLOT:
        model.plot_posterior(ci=0.95)
        plt.suptitle('Linear Regression (flipout est) ' + 
                     'weights=' + str(true_weight) +
                     ', bias=' + str(true_bias) +
                     ', std_err=' + str(true_std_err))
        plt.show()


# TODO: test w/ Dense


# TODO: test w/ LinearRegression


def test_LR_pandas():
    """Tests regression w/ pandas works correctly"""

    # Parameters + input data is vector of length 3
    Nd = 3

    # Model = linear regression assuming error = 1
    weight = Parameter(name='aic_pd_weight', shape=2, estimator=None)
    data = Input(cols=['d', 'b'])
    data2 = Input(cols='c')
    model = Normal(Dot(data, weight) + data2, 1.0)

    # Generate data
    true_weight = np.array([0.5, -0.25])
    noise = np.random.randn(N)
    x = np.random.randn(N, Nd+1)
    x[:,0] = np.sum(true_weight*x[:,[3, 1]], axis=1) + x[:,2] + noise
    df = pd.DataFrame(x, columns=['a', 'b', 'c', 'd'])

    # Fit the model
    model.fit(['b', 'c', 'd'], 'a', data=df, epochs=epochs)

    # Check that the inferences are in the correct ballpark
    means = model.posterior_mean()
    assert abs(means['aic_pd_weight'][0]-0.5) < 0.2
    assert abs(means['aic_pd_weight'][1]+0.25) < 0.2


if __name__ == "__main__":
    PLOT = True
    import matplotlib.pyplot as plt
    test_LR_scalar_no_variance()
    test_LR_scalar()
    test_LR_vector()
    test_LR_vector_flipout()