
import numpy as np
import matplotlib.pyplot as plt
from probflow import *

NUM_SAMPLES = 1000


def test_posterior_plot_scalar(LR1_novar, plot):
    """Tests Parameter.posterior_plot and BaseDistribution.posterior_plot"""

    model = LR1_novar #fixture from conftest.py

    # Plot all the model's posteriors w/ [hist, 1col, no_ci]
    model.posterior_plot(style='hist', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2cols, no_ci]
    model.posterior_plot(style='hist', cols=2, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 1col, ci]
    model.posterior_plot(style='hist', ci=0.9, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2cols, ci]
    model.posterior_plot(style='hist', cols=2, ci=0.9, color='r',
                         num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [line, 1col, no_ci]
    model.posterior_plot(style='line', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [line, 1col, ci]
    model.posterior_plot(style='line', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [fill, 1col, no_ci]
    model.posterior_plot(style='fill', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [fill, 1col, ci]
    model.posterior_plot(style='fill', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot just the weight's posterior w/ a teensy bandwidth
    model.posterior_plot(style='fill', bw=0.01, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot just the bias' posterior w/ a yuge bandwidth
    model.posterior_plot(style='fill', bw=1, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Default should be fill, 1col, no ci
    model.posterior_plot(num_samples=NUM_SAMPLES)
    if plot:
        plt.show()


def test_posterior_plot_vector(LR3_var, plot):
    """Tests Parameter.posterior_plot and BaseDistribution.posterior_plot"""

    model = LR3_var #fixture from conftest.py

    # Plot all the model's posteriors w/ [hist, 1col, no_ci]
    model.posterior_plot(style='line', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2col, no_ci]
    model.posterior_plot(cols=2, ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()
