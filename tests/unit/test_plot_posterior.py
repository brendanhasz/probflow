
import numpy as np
from probflow import *

PLOT = False
epochs = 1
N = 10


def test_plot_posterior_scalar():
    """Tests Parameter.plot_posterior and BaseDistribution.plot_posterior"""

    # Linear regression w/ 2 scalar variables
    weight = Parameter(name='the_weight', estimator='flipout')
    bias = Parameter(name='the_bias', estimator='flipout')
    model = Normal(Input()*weight + bias, 1.0)

    true_weight = 0.5
    true_bias = -1.0

    x = np.linspace(-5, 5, N)
    y = true_weight*x + true_bias + np.random.randn(N)

    model.fit(x, y, epochs=epochs)

    # Plot all the model's posteriors w/ [hist, 1col, no_ci]
    model.plot_posterior(style='hist')
    if PLOT:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2cols, no_ci]
    model.plot_posterior(style='hist', cols=2)
    if PLOT:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 1col, ci]
    model.plot_posterior(style='hist', ci=0.9)
    if PLOT:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2cols, ci]
    model.plot_posterior(style='hist', cols=2, ci=0.9, color='r')
    if PLOT:
        plt.show()

    # Plot all the model's posteriors w/ [line, 1col, no_ci]
    model.plot_posterior(style='line')
    if PLOT:
        plt.show()

    # Plot all the model's posteriors w/ [line, 1col, ci]
    model.plot_posterior(style='line', ci=0.95)
    if PLOT:
        plt.show()

    # Plot all the model's posteriors w/ [fill, 1col, no_ci]
    model.plot_posterior(style='fill')
    if PLOT:
        plt.show()

    # Plot all the model's posteriors w/ [fill, 1col, ci]
    model.plot_posterior(style='fill', ci=0.95)
    if PLOT:
        plt.show()

    # Plot just the weight's posterior w/ a teensy bandwidth
    weight.plot_posterior(style='fill', bw=0.001)
    if PLOT:
        plt.show()

    # Plot just the bias' posterior w/ a yuge bandwidth
    bias.plot_posterior(style='fill', bw=1)
    if PLOT:
        plt.show()

    # Default should be fill, 1col, no ci
    model.plot_posterior()
    if PLOT:
        plt.show()


def test_plot_posterior_vector():
    """Tests Parameter.plot_posterior and BaseDistribution.plot_posterior"""

    Nd = 3

    # Multivariate linear regression
    weight = Parameter(shape=Nd, estimator=None)
    bias = Parameter(estimator=None)
    std_dev = ScaleParameter()
    model = Normal(Dot(Input(), weight) + bias, std_dev)

    # Generate data
    true_weight = np.array([0.5, -0.25, 0.0])
    true_bias = -1.0
    noise = np.random.randn(N, 1)
    x = np.random.randn(N, Nd)
    y = np.expand_dims(np.sum(true_weight*x, axis=1) + true_bias, 1) + noise

    model.fit(x, y, epochs=epochs)

    # Plot all the model's posteriors w/ [hist, 1col, no_ci]
    model.plot_posterior(style='line')
    if PLOT:
        plt.show()

    # Plot all the model's posteriors w/ [hist, 2col, no_ci]
    model.plot_posterior(cols=2, ci=0.95)
    if PLOT:
        plt.show()


if __name__ == "__main__":
    PLOT = True
    import matplotlib.pyplot as plt
    epochs = 1000
    N = 1000
    test_plot_posterior_scalar()
    test_plot_posterior_vector()
