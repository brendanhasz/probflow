"""Tests probflow.utils.plotting module and methods which use it"""



import pytest

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import probflow as pf



def test_approx_kde(plot):
    """Tests utils.plotting.approx_kde"""
    data = np.random.randn(1000)
    x, y = pf.utils.plotting.approx_kde(data)
    assert x.shape[0] == y.shape[0]
    if plot:
        plt.plot(x, y)
        plt.title('should be kde density of samples from norm dist')
        plt.show()



def test_get_next_color():
    """Tests utils.plotting.get_next_color"""
    col = pf.utils.plotting.get_next_color(None, 0)
    assert isinstance(col, str)
    assert col[0] == '#'



def test_get_ix_label():
    """Tests utils.plotting.get_ix_label"""

    # 1d
    lab = pf.utils.plotting.get_ix_label(2, [3])
    assert isinstance(lab, str)
    assert lab == '2'

    # 2d
    lab = pf.utils.plotting.get_ix_label(5, [3, 3])
    assert isinstance(lab, str)
    assert lab == '[2, 1]'

    # 3d
    lab = pf.utils.plotting.get_ix_label(5, [3, 3, 3])
    assert isinstance(lab, str)
    assert lab == '[2, 1, 0]'



def test_plot_dist(plot):
    """Tests utils.plotting.plot_dist"""
    data = np.random.randn(1000)
    pf.utils.plotting.plot_dist(data)
    if plot:
        plt.title('should be kde density (filled) of samples from norm dist')
        plt.show()



def test_posterior_plot(plot):
    """Tests posterior_plot method of parameter and model"""

    class MyModel(pf.Model):

        def __init__(self):
            self.weight = pf.Parameter(name='Weight')
            self.bias = pf.Parameter(name='Bias')
            self.std = pf.ScaleParameter(name='Noise Std Dev')

        def __call__(self, x):
            return pf.Normal(x*self.weight() + self.bias(), self.std())

    # Create the model
    model = MyModel()

    # Plot posterior for a parameter
    model.weight.posterior_plot()
    if plot:
        plt.show()

    # Plot posteriors for all params in the model
    model.posterior_plot()
    if plot:
        plt.show()

    # Should be able to plot just some params
    # and pass kwargs to Parameter.posterior_plot
    model.posterior_plot(params=['Weight', 'Bias'], ci=0.95)
    if plot:
        plt.show()

    # Should be able to change number of columns
    model.posterior_plot(cols=2)
    if plot:
        plt.show()



def test_prior_plot(plot):
    """Tests prior_plot method of parameter and model"""

    class MyModel(pf.Model):

        def __init__(self):
            self.weight = pf.Parameter(name='Weight')
            self.bias = pf.Parameter(name='Bias')
            self.std = pf.ScaleParameter(name='Noise Std Dev')

        def __call__(self, x):
            return pf.Normal(x*self.weight() + self.bias(), self.std())

    # Create the model
    model = MyModel()

    # Plot posterior for a parameter
    model.weight.prior_plot()
    if plot:
        plt.show()

    # Plot posteriors for all params in the model
    model.prior_plot()
    if plot:
        plt.show()

    # Should be able to pass kwargs to posterior_plot
    model.prior_plot(ci=0.95)
    if plot:
        plt.show()

    # Should be able to plot just some params
    model.prior_plot(params=['Weight', 'Bias'], ci=0.95)
    if plot:
        plt.show()

    # Should be able to change number of columns
    model.prior_plot(cols=2)
    if plot:
        plt.show()

