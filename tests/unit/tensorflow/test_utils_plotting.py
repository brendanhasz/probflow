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
        plt.title("should be kde density of samples from norm dist")
        plt.show()


def test_get_next_color():
    """Tests utils.plotting.get_next_color"""

    # default
    col = pf.utils.plotting.get_next_color(None, 0)
    assert isinstance(col, str)
    assert col[0] == "#"

    # list of colors
    col = pf.utils.plotting.get_next_color(["#eeefff", "#gggaaa"], 1)
    assert isinstance(col, str)
    assert col[0] == "#"

    # single color
    col = pf.utils.plotting.get_next_color("#eeefff", 1)
    assert isinstance(col, str)
    assert col[0] == "#"


def test_get_ix_label():
    """Tests utils.plotting.get_ix_label"""

    # 1d
    lab = pf.utils.plotting.get_ix_label(2, [3])
    assert isinstance(lab, str)
    assert lab == "2"

    # 2d
    lab = pf.utils.plotting.get_ix_label(5, [3, 3])
    assert isinstance(lab, str)
    assert lab == "[2, 1]"

    # 3d
    lab = pf.utils.plotting.get_ix_label(5, [3, 3, 3])
    assert isinstance(lab, str)
    assert lab == "[2, 1, 0]"


def test_plot_dist(plot):
    """Tests utils.plotting.plot_dist"""

    data = np.random.randn(1000)

    # Should error on invalid ci
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_dist(data, ci=-0.1)
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_dist(data, ci=1.1)

    pf.utils.plotting.plot_dist(data)
    if plot:
        plt.title("should be kde density (filled) of samples from norm dist")
        plt.show()

    pf.utils.plotting.plot_dist(data, ci=0.9)
    if plot:
        plt.title(
            "should be kde density (filled) of samples from norm dist w/ ci"
        )
        plt.show()

    pf.utils.plotting.plot_dist(data, style="line")
    if plot:
        plt.title("should be line plot of samples from norm dist")
        plt.show()

    pf.utils.plotting.plot_dist(data, style="line", ci=0.9)
    if plot:
        plt.title("should be line plot of samples from norm dist w/ ci")
        plt.show()

    pf.utils.plotting.plot_dist(data, style="hist")
    if plot:
        plt.title("should be line plot of samples from norm dist")
        plt.show()

    pf.utils.plotting.plot_dist(data, style="hist", ci=0.9)
    if plot:
        plt.title("should be line plot of samples from norm dist w/ ci")
        plt.show()

    # Should error on invalid style
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_dist(data, style="lala")

    # Should be able to show multiple distributions
    data = np.random.randn(1000, 3) + np.array([[-2.0, 0.0, 2.0]])
    pf.utils.plotting.plot_dist(data, ci=0.9)
    if plot:
        plt.title("should be 3 kde density (filled) w/ ci")
        plt.show()


def test_plot_line(plot):
    """Tests utils.plotting.plot_line"""

    x = np.linspace(0, 10, 100)
    y = np.random.randn(100)

    # Should error on invalid shapes
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_line(x, np.random.randn(5))

    pf.utils.plotting.plot_line(x, y)
    if plot:
        plt.title("should be noisy line")
        plt.show()

    pf.utils.plotting.plot_line(x, np.random.randn(100, 3))
    if plot:
        plt.title("should be 3 noisy lines w/ labels")
        plt.show()


def test_fill_between(plot):
    """Tests utils.plotting.fill_between"""

    x = np.linspace(0, 10, 100)
    y1 = np.random.randn(100)
    y2 = np.random.randn(100) + 5

    # Should error on invalid shapes
    with pytest.raises(ValueError):
        pf.utils.plotting.fill_between(x, y1, np.random.randn(3))
    with pytest.raises(ValueError):
        pf.utils.plotting.fill_between(np.random.randn(3), y1, y2)

    pf.utils.plotting.fill_between(x, y1, y2)
    if plot:
        plt.title("should be one filled area")
        plt.show()

    y1 = np.random.randn(100, 3)
    y2 = np.random.randn(100, 3) + 3
    y1 += np.array([0, 5, 10])
    y2 += np.array([0, 5, 10])
    pf.utils.plotting.fill_between(x, y1, y2)
    if plot:
        plt.title("should be 3 filled areas")
        plt.show()


def test_centered_text(plot):
    """Tests utils.plotting.centered_text"""
    plt.plot(np.linspace(0, 1, 10), np.random.randn(10))
    pf.utils.plotting.centered_text("lala")
    if plot:
        plt.title("should be fig w/ lala in middle")
        plt.show()


def test_plot_discrete_dist(plot):
    """Tests utils.plotting.plot_discrete_dist"""

    # Should work for categorical variables
    pf.utils.plotting.plot_discrete_dist(np.array([0, 0, 1, 1, 1, 2]))
    if plot:
        plt.title("should be 0 1 2")
        plt.show()

    # Should work for discrete variables
    pf.utils.plotting.plot_discrete_dist(tf.random.poisson([200], 5).numpy())
    if plot:
        plt.title("should be poisson-y")
        plt.show()

    # xlabel shouldn't show ALL values if lots of uniques
    pf.utils.plotting.plot_discrete_dist(tf.random.poisson([2000], 10).numpy())
    if plot:
        plt.title("should be poisson-y")
        plt.show()


def test_plot_categorical_dist(plot):
    """Tests utils.plotting.plot_categorical_dist"""

    # Should work for categorical variables
    pf.utils.plotting.plot_categorical_dist(np.array([0, 0, 1, 1, 1, 2]))
    if plot:
        plt.title("should be 0 1 2")
        plt.show()

    # xlabel shouldn't show ALL values if lots of uniques
    pf.utils.plotting.plot_categorical_dist(
        tf.random.poisson([2000], 50).numpy()
    )
    if plot:
        plt.title("should be poisson-y, not showing all xticklabels")
        plt.show()


def test_plot_by(plot):
    """Tests utils.plotting.plot_by"""

    x = np.linspace(0, 10, 100)
    data = np.random.randn(100)

    # Should error on invalid args
    with pytest.raises(TypeError):
        pf.utils.plotting.plot_by(x, data, bins=0.1)
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_by(x, data, bins=0)
    with pytest.raises(TypeError):
        pf.utils.plotting.plot_by(x, data, plot="asdf")
    with pytest.raises(TypeError):
        pf.utils.plotting.plot_by(x, data, bootstrap="asdf")
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_by(x, data, bootstrap=0)
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_by(x, data, func="asdf")
    with pytest.raises(TypeError):
        pf.utils.plotting.plot_by(x, data, func=0)
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_by(np.random.randn(2, 3, 4), data)
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_by(x, data, ci=-0.1)
    with pytest.raises(ValueError):
        pf.utils.plotting.plot_by(x, data, ci=1.1)

    pf.utils.plotting.plot_by(x, data, func=lambda x: np.mean(x))
    pf.utils.plotting.plot_by(x, data, func="mean", color="#eeefff")
    pf.utils.plotting.plot_by(x, data, func="median")
    pf.utils.plotting.plot_by(x, data, func="count")

    # Should plot mean data by x
    plt.clf()
    pf.utils.plotting.plot_by(x, data)
    if plot:
        plt.title("plot mean(randn) by x")
        plt.show()

    # Should plot 2D plot of mean(data) by x[:, 0] and x[:, 1]
    x = np.random.randn(100, 2)
    pf.utils.plotting.plot_by(x, data)
    if plot:
        plt.title("plot mean(randn) by x")
        plt.show()


def test_posterior_plot(plot):
    """Tests posterior_plot method of parameter and model"""

    class MyModel(pf.Model):
        def __init__(self):
            self.weight = pf.Parameter(name="Weight")
            self.bias = pf.Parameter(name="Bias")
            self.std = pf.ScaleParameter(name="Noise Std Dev")

        def __call__(self, x):
            return pf.Normal(x * self.weight() + self.bias(), self.std())

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
    model.posterior_plot(params=["Weight", "Bias"], ci=0.95)
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
            self.weight = pf.Parameter(name="Weight")
            self.bias = pf.Parameter(name="Bias")
            self.std = pf.ScaleParameter(
                name="Noise Std Dev", prior=pf.Gamma(1.0, 1.0)
            )

        def __call__(self, x):
            return pf.Normal(x * self.weight() + self.bias(), self.std())

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
    model.prior_plot(params=["Weight", "Bias"], ci=0.95)
    if plot:
        plt.show()

    # Should be able to change number of columns
    model.prior_plot(cols=2)
    if plot:
        plt.show()
