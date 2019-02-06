
import numpy as np
import matplotlib.pyplot as plt
from probflow import *

NUM_SAMPLES = 1000


def test_plot_prior_scalar(LR1_novar, plot):
    """Tests Parameter.plot_prior and BaseDistribution.plot_prior"""

    model = LR1_novar #fixture from conftest.py

    # Plot just the weight's prior w/ a teensy bandwidth
    model.plot_prior(style='fill', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot just the bias' prior w/ a yuge bandwidth
    model.plot_prior(style='fill', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 1col, no_ci]
    model.plot_prior(style='hist', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 2cols, no_ci]
    model.plot_prior(style='hist', cols=2, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 1col, ci]
    model.plot_prior(style='hist', ci=0.9, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 2cols, ci]
    model.plot_prior(style='hist', cols=2, ci=0.9, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [line, 1col, no_ci]
    model.plot_prior(style='line', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [line, 1col, ci]
    model.plot_prior(style='line', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [fill, 1col, no_ci]
    model.plot_prior(style='fill', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [fill, 1col, ci]
    model.plot_prior(style='fill', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Default should be fill, 1col, no ci
    model.plot_prior(num_samples=NUM_SAMPLES)
    if plot:
        plt.show()


def test_plot_prior_vector(LR3_var, plot):
    """Tests Parameter.plot_prior and BaseDistribution.plot_prior"""

    model = LR3_var #fixture from conftest.py

    # Plot all the model's priors w/ [hist, 1col, no_ci]
    model.plot_prior(style='line', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 2col, no_ci]
    model.plot_prior(style='fill', ci=0.95, cols=2, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()
