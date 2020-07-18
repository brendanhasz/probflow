import probflow.utils.ops as O
from probflow.distributions import Normal
from probflow.utils.casting import to_numpy
from probflow.utils.initializers import scale_xavier, xavier

from .parameter import Parameter


class BoundedParameter(Parameter):
    r"""A parameter bounded on either side

    This is a convenience class for creating a parameter :math:`\beta` bounded
    on both sides.  It uses a logit-normal posterior distribution:

    .. math::

        \text{Logit}(\beta) = \log \left( \frac{\beta}{1-\beta} \right)
            \sim \text{Normal}(\mu, \sigma)


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
        Default is to use a sigmoid transform.
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
        Default = ``'BoundedParameter'``

    Examples
    --------

    TODO

    """

    def __init__(
        self,
        shape=1,
        posterior=Normal,
        prior=Normal(0, 1),
        transform=None,
        initializer={"loc": xavier, "scale": scale_xavier},
        var_transform={"loc": None, "scale": O.softplus},
        min: float = 0.0,
        max: float = 1.0,
        name="BoundedParameter",
    ):

        # Check bounds
        if min > max:
            raise ValueError("min is larger than max")

        # Create the transform based on the bounds
        if transform is None:
            transform = lambda x: min + (max - min) * O.sigmoid(x)

        # Create the parameter
        super().__init__(
            shape=shape,
            posterior=posterior,
            prior=prior,
            transform=transform,
            initializer=initializer,
            var_transform=var_transform,
            name=name,
        )
