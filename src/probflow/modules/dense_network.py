from typing import Callable, List, Union

import probflow.utils.ops as O

from .dense import Dense
from .module import Module


class DenseNetwork(Module):
    r"""A multilayer dense neural network

    TODO: explain, math, diagram, examples, etc

    Parameters
    ----------
    d : List[int]
        Dimensionality (number of units) for each layer.
        The first element should be the dimensionality of the independent
        variable (number of features).
    activation : callable
        Activation function to apply to the outputs of each layer.
        Note that the activation function will not be applied to the outputs
        of the final layer.
        Default = :math:`\max ( 0, x )`

    Attributes
    ----------
    layers : List[:class:`.Dense`]
        List of :class:`.Dense` neural network layers to be applied
    activations : List[callable]
        Activation function for each layer
    """

    def __init__(
        self,
        d: List[int],
        activation: Callable = O.relu,
        name: str = "DenseNetwork",
    ):
        self.activations = [activation for i in range(len(d) - 2)]
        self.activations += [lambda x: x]
        self.name = name
        names = [name + "_Dense" + str(i) for i in range(len(d) - 1)]
        self.layers = [
            Dense(d[i], d[i + 1], name=names[i]) for i in range(len(d) - 1)
        ]

    def __call__(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x
