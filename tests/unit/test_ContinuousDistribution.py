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
