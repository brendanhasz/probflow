from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class MultivariateNormal(BaseDistribution):
    r"""The multivariate Normal distribution.

    The
    `multivariate normal distribution <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_
    is a continuous distribution in :math:`d`-dimensional space, and has two
    parameters:

    - a location vector (``loc`` or :math:`\boldsymbol{\mu} \in \mathbb{R}^d`)
      which determines the mean of the distribution, and
    - a covariance matrix (``scale`` or
      :math:`\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}_{>0}`) which
      determines the spread and covariance of the distribution.

    A random variable :math:`\mathbf{x} \in \mathbb{R}^d` drawn from a
    multivariate normal distribution

    .. math::

        \mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})

    has probability

    .. math::

        p(\mathbf{x}) = (2\pi)^{-\frac{d}{2}}
            \det(\boldsymbol{\Sigma})^{-\frac{1}{2}}
            \exp \left(
                -\frac{1}{2}
                (\mathbf{x}-\boldsymbol{\mu})^\top
                \boldsymbol{\Sigma}^{-1}
                (\mathbf{x}-\boldsymbol{\mu})
            \right)

    TODO: example image of the distribution


    Parameters
    ----------
    loc : |ndarray|, or Tensor
        Mean of the multivariate normal distribution
        (:math:`\boldsymbol{\mu}`).
    cov : |ndarray|, or Tensor
        Covariance matrix of the multivariate normal distribution
        (:math:`\boldsymbol{\Sigma}`).
    """

    def __init__(self, loc, cov):

        # Check input
        ensure_tensor_like(loc, "loc")
        ensure_tensor_like(cov, "cov")

        # Store args
        self.loc = loc
        self.cov = cov

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.multivariate_normal.MultivariateNormal(
                self["loc"], covariance_matrix=self["cov"]
            )
        else:
            import tensorflow as tf
            from tensorflow_probability import distributions as tfd

            tril = tf.linalg.cholesky(self["cov"])
            return tfd.MultivariateNormalTriL(loc=self["loc"], scale_tril=tril)
