from typing import Callable, List

import probflow.utils.ops as O

from .batch_normalization import BatchNormalization
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
    probabilistic : bool
        Whether variational posteriors for the weights and biases should be
        probabilistic.  If True (the default), will use Normal distributions
        for the variational posteriors.  If False, will use Deterministic
        distributions.
    batch_norm : bool
        Whether or not to use batch normalization in between layers of the
        network.  Default is False.
    batch_norm_loc : str {'after' or 'before'}
        Where to apply the batch normalization.  If ``'after'``, applies the
        batch normalization after the activation.  If ``'before'``, applies the
        batch normalization before the activation.  Default is ``'after'``.
    batch_norm_kwargs : dict
        Additional parameters to pass to :class:`.BatchNormalization` for each
        layer.
    kwargs
        Additional parameters are passed to :class:`.Dense` for each layer.

    Attributes
    ----------
    layers : List[:class:`.Dense`]
        List of :class:`.Dense` neural network layers to be applied
    activations : List[callable]
        Activation function for each layer
    batch_norms : Union[None, List[:class:`.BatchNormalization`]]
        Batch normalization layers
    """

    def __init__(
        self,
        d: List[int],
        activation: Callable = O.relu,
        batch_norm: bool = False,
        batch_norm_loc: str = "after",
        name: str = "DenseNetwork",
        batch_norm_kwargs: dict = {},
        **kwargs
    ):

        self.name = name

        # Activations
        self.activations = [activation for i in range(len(d) - 2)]
        self.activations += [lambda x: x]

        # Dense layers
        names = [name + "_Dense" + str(i) for i in range(len(d) - 1)]
        self.layers = [
            Dense(d[i], d[i + 1], name=names[i], **kwargs)
            for i in range(len(d) - 1)
        ]

        # Batch normalization
        self.batch_norm = batch_norm
        self.batch_norm_loc = batch_norm_loc
        if batch_norm:
            names = [
                name + "_BatchNormalization" + str(i)
                for i in range(len(d) - 2)
            ]
            self.batch_norms = [
                BatchNormalization(
                    d[i + 1], name=names[i], **batch_norm_kwargs
                )
                for i in range(len(d) - 2)
            ]
            self.batch_norms += [lambda x: x]  # no batch norm after last layer

    def __call__(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if self.batch_norm and self.batch_norm_loc == "before":
                x = self.batch_norms[i](x)
            x = self.activations[i](x)
            if self.batch_norm and self.batch_norm_loc == "after":
                x = self.batch_norms[i](x)
        return x
