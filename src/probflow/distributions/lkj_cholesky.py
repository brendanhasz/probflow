"""The LKJ distribution for lower Cholesky factors of correlation matrices."""

from probflow.utils.base import BaseDistribution
from probflow.utils.settings import ProbflowBackend, get_backend
from probflow.utils.typing import BackendDistribution, TensorLike
from probflow.utils.validation import ensure_tensor_like


class LKJCholesky(BaseDistribution):
    r"""The LKJ distribution for lower Cholesky factors of correlation matrices.

    The `LKJ distribution <https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution>`_
    is a distribution over positive definite correlation matrices with
    diagonals = 1 (:math:`\boldsymbol{\Sigma}`, a :math:`d \times d` matrix).
    The Cholesky parameterization models the correlation matrix indirectly by
    specifying a distribution over its lower Cholesky factor :math:`L` (a
    :math:`d \times d` lower triangular matrix), where
    :math:`\boldsymbol{\Sigma} = L L^\top`.
    
    It has two arguments:
    
    - ``dim``: the number of dimensions of the correlation matrix (:math:`d`), and
    - ``eta``: a concentration parameter (:math:`\eta > 0`) which
      determines the concentration of the distribution around the identity matrix.
      Values for :math:`\eta` greater than 1 result in correlation matrices concentrated
      more closely around the identity matrix, while values less than 1 allow for more
      diffuse correlations.


    Parameters
    ----------
    dim : int
        The number of dimensions of the correlation matrix.
    eta : float or |ndarray| or Tensor
        The concentration parameter of the LKJ distribution.
    """

    def __init__(self, dim: int, eta: TensorLike) -> None:

        # Check input
        assert dim > 0, "dim must be a positive integer"
        ensure_tensor_like(eta, "eta")

        # Store args
        self.dim = dim
        self.eta = eta

    def __call__(self) -> BackendDistribution:
        """Get the distribution object from the backend."""
        if get_backend() == ProbflowBackend.PYTORCH:
            import torch.distributions as tod

            return tod.lkj_cholesky.LKJCholesky(
                dim=self.dim, concentration=self.eta
            )
        elif get_backend() == ProbflowBackend.JAX:
            from tensorflow_probability.substrates import jax as tfp_jax

            return tfp_jax.distributions.CholeskyLKJ(
                dimension=self.dim, eta=self.eta
            )
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.CholeskyLKJ(dimension=self.dim, concentration=self.eta)
