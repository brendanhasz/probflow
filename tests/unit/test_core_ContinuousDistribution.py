"""Tests probflow.core.ContinuousDistribution class"""

import pytest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from probflow import *

EPOCHS = 2
N = 10



def test_ContinuousDistribution_predictive_distribution_plot(LR3_novar, plot):
    """Tests core.ContinuousDistribution.predictive_distribution_plot"""

    model = LR3_novar #fixture from conftest.py
    x_val = np.random.rand(10, 3)

    # Check predictive_distribution with no input
    prd = model.predictive_distribution_plot(x_val, style='line')
    if plot:
        plt.suptitle('should show 10 line dists')
        plt.show()

    # Check predictive_distribution with no input
    prd = model.predictive_distribution_plot(x_val, individually=True, cols=2)
    if plot:
        plt.suptitle('should show 5x2 grid of 10 fill dists')
        plt.tight_layout()
        plt.show()

    # Check predictive_distribution with validation input
    x_val = np.random.rand(1, 3)
    prd = model.predictive_distribution_plot(x_val)
    if plot:
        plt.suptitle('should show a single fill dist')
        plt.show()

    # Check predictive_distribution with conf intervals
    prd = model.predictive_distribution_plot(x_val, ci=0.95)
    if plot:
        plt.suptitle('should show a single fill dist w/ 95prc ci')
        plt.show()



def test_ContinuousDistribution_coverage_by(plot):
    """Tests core.test_ContinuousDistribution.coverage_by"""

    # Dummy data
    x = np.random.rand(N)
    true_noise = 0.2
    y = x + true_noise*np.random.randn(N)

    # Add extra noise which model shouldn't capture well
    ix = (x>0.2) & (x<0.4)
    y[ix] = y[ix] + 0.4*np.random.randn(np.count_nonzero(ix))

    # ProbFlow model
    model = Normal(Parameter()*Input()+Parameter(), 0.18)
    model.fit(x, y, epochs=EPOCHS)

    # Check coverage_by works
    model.coverage_by()
    if plot:
        plt.xlabel('Value of x')
        plt.tight_layout()
        plt.show()
