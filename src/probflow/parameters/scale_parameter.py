import probflow.utils.ops as O
from probflow.distributions import Gamma
from probflow.utils.initializers import full_of

from .parameter import Parameter


class ScaleParameter(Parameter):
    r"""Standard deviation parameter.

    This is a convenience class for creating a standard deviation parameter
    (:math:`\sigma`).  It is created by first constructing a variance
    parameter (:math:`\sigma^2`) which uses an inverse gamma distribution as
    the variational posterior.

    .. math::

        \frac{1}{\sigma^2} \sim \text{Gamma}(\alpha, \beta)

    Then the variance is transformed into the standard deviation:

    .. math::

        \sigma = \sqrt{\sigma^2}

    By default, an inverse gamma prior is used:

    .. math::

        \frac{1}{\sigma^2} \sim \text{Gamma}(5, 5)


    Parameters
    ----------
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Gamma`
    prior : |Distribution| object or |None|
        Prior probability distribution function which has been instantiated
        with parameters, or |None| for a uniform prior.
        Default = ``None``
    transform : callable
        Transform to apply to the random variable.
        Default is to use an inverse square root transform (``sqrt(1/x)``)
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'ScaleParameter'``

    Examples
    --------

    Use :class:`.ScaleParameter` to create a standard deviation parameter
    for a :class:`.Normal` distribution:

    TODO

    """

    def __init__(
        self,
        shape=1,
        posterior=Gamma,
        prior=Gamma(5, 1),
        transform=lambda x: O.sqrt(1.0 / x),
        initializer={"concentration": full_of(4.0), "rate": full_of(1.0)},
        var_transform={"concentration": O.exp, "rate": O.exp},
        name="ScaleParameter",
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
