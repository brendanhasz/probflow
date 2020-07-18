from typing import List, Union

import probflow.utils.ops as O
from probflow.distributions import Dirichlet
from probflow.utils.initializers import pos_xavier

from .parameter import Parameter


class DirichletParameter(Parameter):
    r"""Dirichlet parameter.

    This is a convenience class for creating a parameter
    :math:`\theta` with a Dirichlet posterior:

    .. math::

        \theta \sim \text{Dirichlet}(\mathbf{\alpha})

    By default, a uniform Dirichlet prior is used:

    .. math::

        \theta \sim \text{Dirichlet}_K(\mathbf{1}/K)

    TODO: explain that a sample is a categorical prob dist (as compared to
    CategoricalParameter, where a sample is a single value)


    Parameters
    ----------
    k : int > 2
        Number of categories.
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Dirichlet`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Dirichlet` ``(1)``
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
        Default = ``'DirichletParameter'``


    Examples
    --------

    TODO: creating variable

    """

    def __init__(
        self,
        k: int = 2,
        shape: Union[int, List[int]] = [],
        posterior=Dirichlet,
        prior=None,
        transform=None,
        initializer={"concentration": pos_xavier},
        var_transform={"concentration": O.softplus},
        name="DirichletParameter",
    ):

        # Check type of k
        if not isinstance(k, int):
            raise TypeError("k must be an integer")
        if k < 2:
            raise ValueError("k must be >1")

        # Make shape a list
        if isinstance(shape, int):
            shape = [shape]

        # Create shape of underlying variable array
        shape = shape + [k]

        # Use a uniform prior
        if prior is None:
            prior = Dirichlet(O.ones(shape))

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
