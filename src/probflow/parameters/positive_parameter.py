"""A parameter which takes only positive values."""

from collections.abc import Callable

import probflow.utils.ops as O
from probflow.distributions import Normal
from probflow.parameters.parameter import Parameter
from probflow.utils.base import BaseDistribution
from probflow.utils.initializers import scale_xavier, xavier


class PositiveParameter(Parameter):
    r"""A parameter which takes only positive values.

    This is a convenience class for creating a parameter :math:`\beta` which
    can only take positive values.  It uses a normal variational posterior
    distribution and a softplus transform:

    .. math::

        \log ( 1 + \exp ( \beta )) \sim \text{Normal}(\mu, \sigma)


    Parameters
    ----------
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0, 1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use a softplus transform.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    min : float
        Minimum value the parameter can take.
        Default = 0.
    max : float
        Maximum value the parameter can take.
        Default = 1.
    name : str
        Name of the parameter(s).
        Default = ``'PositiveParameter'``

    Examples
    --------
    TODO

    """

    def __init__(
        self,
        shape: int | list[int] = 1,
        posterior: type[BaseDistribution] = Normal,
        prior: BaseDistribution = Normal(0, 1),
        transform: Callable | None = O.softplus,
        initializer: dict[str, Callable] = {
            "loc": xavier,
            "scale": scale_xavier,
        },
        var_transform: dict[str, Callable | None] = {
            "loc": None,
            "scale": O.softplus,
        },
        name: str = "PositiveParameter",
    ):
        super().__init__(
            shape=shape,
            posterior=posterior,
            prior=prior,
            transform=transform,
            initializer=initializer,
            var_transform=var_transform,
            name=name,
        )
