"""A parameter with a Wishart posterior + prior."""

import probflow.utils.ops as O
from probflow.distributions import Wishart
from probflow.parameters.parameter import Parameter
from probflow.utils.base import BaseDistribution
from probflow.utils.initializers import xavier


class WishartParameter(Parameter):
    r"""A parameter with a Wishart posterior + prior.

    Uses the Cholesky decomposition lower triangular matrix of the scale
    matrix for parameterization.  To have the parameter samples be full
    covariance matrices, set ``output_tril=False``.

    TODO: math, explanations of different defn's via last 3 kwargs


    Parameters
    ----------
    d : int
        Number of dimensions
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Wishart` ``(d+1, I(d, d))``
    expand_dims : int or None
        Dimension to expand output samples along.
    name : str
        Name of the parameter(s).
        Default = ``'WishartParameter'``

    Examples
    --------
    TODO

    References
    ----------
    - Jose C. Pinheiro & Douglas M. Bates.
      `Unconstrained Parameterizations for Variance-Covariance Matrices <https://dx.doi.org/10.1007/BF00140873>`_
      *Statistics and Computing*, 1996.
    """

    def __init__(
        self,
        d: int,
        prior: BaseDistribution | None = None,
        output_tril: bool = True,
        #expand_dims: int = -1,
        name: str = "WishartParameter",
    ):

        # Prior
        if prior is None:
            prior = Wishart(d + 1, O.eye(d))

        # Transform
        if output_tril:
            transform = None
        else:
            transform = lambda x: x @ O.transpose(x)

        # Initializer and variable transforms
        initializer = {
            "df": lambda x: xavier([1]),
            "scale_tril": lambda x: xavier([int(d * (d + 1) / 2)]),
        }
        var_transform = {
            "df": lambda x: d + tf.exp(x),
            "scale_tril": O.log_cholesky_transform,
        }

        super().__init__(
            posterior=Wishart,
            prior=prior,
            transform=transform,
            initializer=initializer,
            var_transform=var_transform,
            name=name,
        )
