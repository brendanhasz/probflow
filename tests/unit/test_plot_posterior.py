
import numpy as np
import matplotlib.pyplot as plt
from probflow import *

NUM_SAMPLES = 1000


def test_plot_posterior_scalar(LR1_novar, plot):
    """Tests Parameter.plot_posterior and BaseDistribution.plot_posterior"""

    model = LR1_novar #fixture from conftest.py

    # Plot all the model's posteriors w/ [hist, 1col, no_ci]
    model.plot_posterior(style='hist', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2cols, no_ci]
    model.plot_posterior(style='hist', cols=2, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 1col, ci]
    model.plot_posterior(style='hist', ci=0.9, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2cols, ci]
    model.plot_posterior(style='hist', cols=2, ci=0.9, color='r',
                         num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [line, 1col, no_ci]
    model.plot_posterior(style='line', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [line, 1col, ci]
    model.plot_posterior(style='line', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [fill, 1col, no_ci]
    model.plot_posterior(style='fill', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [fill, 1col, ci]
    model.plot_posterior(style='fill', ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot just the weight's posterior w/ a teensy bandwidth
    model.plot_posterior(style='fill', bw=0.01, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot just the bias' posterior w/ a yuge bandwidth
    model.plot_posterior(style='fill', bw=1, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Default should be fill, 1col, no ci
    model.plot_posterior(num_samples=NUM_SAMPLES)
    if plot:
        plt.show()


def test_plot_posterior_vector(LR3_var, plot):
    """Tests Parameter.plot_posterior and BaseDistribution.plot_posterior"""

    model = LR3_var #fixture from conftest.py

    # Plot all the model's posteriors w/ [hist, 1col, no_ci]
    model.plot_posterior(style='line', num_samples=NUM_SAMPLES)
    if plot:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2col, no_ci]
    model.plot_posterior(cols=2, ci=0.95, num_samples=NUM_SAMPLES)
    if plot:
        plt.show()
