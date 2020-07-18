import probflow.utils.ops as O
from probflow.distributions import Deterministic, Normal
from probflow.utils.initializers import xavier

from .parameter import Parameter


class DeterministicParameter(Parameter):
    r"""A parameter which takes only a single value (i.e., the posterior is a
    single point value, not a probability distribution).


    Parameters
    ----------
    shape : int or List[int]
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Deterministic`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0, 1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use no transformation.
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'PositiveParameter'``

    Examples
    --------

    TODO

    """

    def __init__(
        self,
        shape=1,
        posterior=Deterministic,
        prior=Normal(0, 1),
        transform=None,
        initializer={"loc": xavier},
        var_transform={"loc": None},
        name="DeterministicParameter",
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
