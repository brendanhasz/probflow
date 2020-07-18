from typing import List, Union

import probflow.utils.ops as O
from probflow.distributions import Categorical
from probflow.utils.initializers import xavier

from .parameter import Parameter


class CategoricalParameter(Parameter):
    r"""Categorical parameter.

    This is a convenience class for creating a categorical parameter
    :math:`\beta` with a Categorical posterior:

    .. math::

        \beta \sim \text{Categorical}(\mathbf{\theta})

    By default, a uniform prior is used.

    TODO: explain that a sample is an int in [0, k-1]


    Parameters
    ----------
    k : int > 2
        Number of categories.
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Categorical`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Categorical` ``(1/k)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use no transform.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'CategoricalParameter'``


    Examples
    --------

    TODO: creating variable

    """

    def __init__(
        self,
        k: int = 2,
        shape: Union[int, List[int]] = [],
        posterior=Categorical,
        prior=None,
        transform=None,
        initializer={"probs": xavier},
        var_transform={"probs": O.additive_logistic_transform},
        name="CategoricalParameter",
    ):

        # Check type of k
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 2:
            raise ValueError("k must be >1")

        # Make shape a list
        if isinstance(shape, int):
            shape = [shape]

        # Use a uniform prior
        if prior is None:
            prior = Categorical(O.ones(shape) / float(k))

        # Create shape of underlying variable array
        shape = shape + [k - 1]

        # Initialize the parameter
        super().__init__(
            shape=shape,
            posterior=posterior,
            prior=prior,
            transform=transform,
            initializer=initializer,
            var_transform=var_transform,
            name=name,
        )

        # shape should correspond to the sample shape
        self.shape = shape
