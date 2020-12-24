import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from probflow.distributions import Normal, Poisson
from probflow.models import DiscreteModel
from probflow.parameters import Parameter


def test_DiscreteModel(plot):
    """Tests probflow.models.DiscreteModel"""

    class MyModel(DiscreteModel):
        def __init__(self):
            self.weight = Parameter([5, 1], name="Weight")
            self.bias = Parameter([1, 1], name="Bias")
            self.act = torch.nn.Softplus()

        def __call__(self, x):
            return Poisson(self.act(x @ self.weight() + self.bias()))

    # Instantiate the model
    model = MyModel()

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = np.round(np.exp(x @ w + 1))

    # Fit the model
    model.fit(x, y, batch_size=50, epochs=100, lr=0.1)

    # plot the predictive dist
    model.pred_dist_plot(x[:1, :])
    if plot:
        plt.title("should be one discrete dist")
        plt.show()

    model.pred_dist_plot(x[:3, :])
    if plot:
        plt.title("should be three discrete dists")
        plt.show()

    model.pred_dist_plot(x[:3, :], cols=2)
    if plot:
        plt.title("should be three discrete dists, two cols")
        plt.show()

    # r_squared shouldn't work!
    with pytest.raises(RuntimeError):
        model.r_squared(x)

    # r_squared shouldn't work!
    with pytest.raises(RuntimeError):
        model.r_squared_plot(x)

    class MyModel(DiscreteModel):
        def __init__(self):
            self.weight = Parameter([5, 1], name="Weight")
            self.bias = Parameter([1, 1], name="Bias")

        def __call__(self, x):
            return Normal(x, 1.0)

    # Instantiate the model
    model = MyModel()

    # Shouldn't work with non-discrete/scalar outputs
    with pytest.raises(NotImplementedError):
        model.pred_dist_plot(x[:1, :])
