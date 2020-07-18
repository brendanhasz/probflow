import probflow.utils.ops as O
from probflow.distributions import MultivariateNormal
from probflow.utils.initializers import xavier
from probflow.utils.settings import get_backend

from .parameter import Parameter


class MultivariateNormalParameter(Parameter):
    r"""A parameter with a multivariate normal posterior, with full covariance.

    TODO: uses the log-Cholesky parameterization (Pinheiro & Bates, 1996).

    TODO: support shape?

    Parameters
    ----------
    d : int
        Number of dimensions
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.MultivariateNormal` ``(0, I)``
    expand_dims : int or None
        Dimension to expand output samples along.
    name : str
        Name of the parameter(s).
        Default = ``'MultivariateNormalParameter'``

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
        d: int = 1,
        prior=None,
        expand_dims: int = -1,
        name="MultivariateNormalParameter",
    ):

        # Transformation for scale parameters
        def log_cholesky_transform(x):
            if get_backend() == "pytorch":
                raise NotImplementedError
            else:
                import tensorflow as tf
                import tensorflow_probability as tfp

                E = tfp.math.fill_triangular(x)
                E = tf.linalg.set_diag(
                    E, tf.exp(tf.linalg.tensor_diag_part(E))
                )
                return E @ tf.transpose(E)

        # Prior
        if prior is None:
            prior = MultivariateNormal(O.zeros([d]), O.eye(d))

        # Transform
        if expand_dims is not None:
            transform = lambda x: O.expand_dims(x, expand_dims)
        else:
            transform = None

        # Initializer and variable transforms
        initializer = {
            "loc": lambda x: xavier([d]),
            "cov": lambda x: xavier([int(d * (d + 1) / 2)]),
        }
        var_transform = {"loc": None, "cov": log_cholesky_transform}

        super().__init__(
            posterior=MultivariateNormal,
            prior=prior,
            transform=transform,
            initializer=initializer,
            var_transform=var_transform,
            name=name,
        )
