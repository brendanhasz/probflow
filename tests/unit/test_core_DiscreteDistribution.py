"""Tests probflow.core.ContinuousDistribution class"""

import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *

N = 10
D = 4
EPOCHS = 2



def test_DiscreteDistribution_predictive_distribution_plot(plot):
    """Tests core.DiscreteDistribution.predictive_distribution_plot"""


    # Dummy data
    x = np.random.randn(N, D)
    x_val1 = np.random.randn(1, D)
    x_val10 = np.random.randn(10, D)
    w = np.random.randn(1, D)
    b = np.random.randn()
    noise = np.random.randn(N)
    y = np.round(1.0/(1.0 + np.exp(-(np.sum(x*w, axis=1) + b + noise))))

    # Logistic regression model
    weights = Parameter(shape=D)
    bias = Parameter()
    logits = Dot(Input(), weights) + bias
    model = Bernoulli(logits)

    # Fit the model
    model.fit(x, y, epochs=EPOCHS)

    # Check predictive_distribution with single input
    prd = model.predictive_distribution_plot(x_val1)
    if plot:
        plt.suptitle('should show a bar @ 0 and a bar @ 1')
        plt.show()

    # Check predictive_distribution with several inputs
    prd = model.predictive_distribution_plot(x_val10)
    if plot:
        plt.suptitle('should show 10 vertically stacked subplots')
        plt.show()

    # Check predictive_distribution with several inputs
    prd = model.predictive_distribution_plot(x_val10, cols=2)
    if plot:
        plt.suptitle('should show 5x2 subplots')
        plt.show()
