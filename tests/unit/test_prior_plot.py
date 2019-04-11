
import numpy as np
import matplotlib.pyplot as plt
from probflow import *

NUM_SAMPLES = 1000


def test_prior_plot_scalar(LR1_novar, plot):
    """Tests Parameter.prior_plot and BaseDistribution.prior_plot"""

    model = LR1_novar #fixture from conftest.py

    # Plot just the weight's prior w/ a teensy bandwidth
    model.prior_plot(style='fill', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot just the bias' prior w/ a yuge bandwidth
    model.prior_plot(style='fill', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 1col, no_ci]
    model.prior_plot(style='hist', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 2cols, no_ci]
    model.prior_plot(style='hist', cols=2, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 1col, ci]
    model.prior_plot(style='hist', ci=0.9, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 2cols, ci]
    model.prior_plot(style='hist', cols=2, ci=0.9, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [line, 1col, no_ci]
    model.prior_plot(style='line', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [line, 1col, ci]
    model.prior_plot(style='line', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [fill, 1col, no_ci]
    model.prior_plot(style='fill', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [fill, 1col, ci]
    model.prior_plot(style='fill', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Default should be fill, 1col, no ci
    model.prior_plot(num_samples=NUM_SAMPLES)
    if plot:
        plt.show()


def test_prior_plot_vector(LR3_var, plot):
    """Tests Parameter.prior_plot and BaseDistribution.prior_plot"""

    model = LR3_var #fixture from conftest.py

    # Plot all the model's priors w/ [hist, 1col, no_ci]
    model.prior_plot(style='line', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's priors w/ [hist, 2col, no_ci]
    model.prior_plot(style='fill', ci=0.95, cols=2, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()
