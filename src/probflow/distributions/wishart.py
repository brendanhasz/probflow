"""The Wishart distribution."""

from probflow.utils.base import BaseDistribution
from probflow.utils.settings import ProbflowBackend, get_backend
from probflow.utils.typing import BackendDistribution, TensorLike
from probflow.utils.validation import ensure_tensor_like


class Wishart(BaseDistribution):
    r"""The Wishart distribution.

    The
    `Wishart distribution <https://en.wikipedia.org/wiki/Wishart_distribution>`_
    is a distribution over symmetric, positive definite matrices.

    TODO: math, explanations of different defn's via last 3 kwargs


    Parameters
    ----------
    df : float or |ndarray| or Tensor
        Degrees of freedom of the Wishart distribution.
    covariance_matrix : |ndarray|, or Tensor
        Covariance matrix of the Wishart distribution.
    precision_matrix : |ndarray|, or Tensor
        Precision matrix of the Wishart distribution.
    scale_tril : |ndarray|, or Tensor
        Lower triangular matrix of the Cholesky decomposition of the covariance
        matrix of the Wishart distribution.
    """

    def __init__(
        self,
        df: TensorLike,
        covariance_matrix: TensorLike | None = None,
        precision_matrix: TensorLike | None = None,
        scale_tril: TensorLike | None = None,
    ) -> None:

        # Check input
        ensure_tensor_like(df, "df")
        if covariance_matrix is None and precision_matrix is None and scale_tril is None:
            raise ValueError("At least one of covariance_matrix, precision_matrix, or scale_tril must be provided.")
        if covariance_matrix is not None:
            ensure_tensor_like(covariance_matrix, "covariance_matrix")
        if precision_matrix is not None:
            ensure_tensor_like(precision_matrix, "precision_matrix")
        if scale_tril is not None:
            ensure_tensor_like(scale_tril, "scale_tril")

        # Store args
        self.df = df
        self.covariance_matrix = covariance_matrix
        self.precision_matrix = precision_matrix
        self.scale_tril = scale_tril

    def __call__(self) -> BackendDistribution:
        """Get the distribution object from the backend."""
        if get_backend() == ProbflowBackend.PYTORCH:
            import torch.distributions as tod

            return tod.wishart.Wishart(
                df=self.df,
                covariance_matrix=self.covariance_matrix,
                precision_matrix=self.precision_matrix,
                scale_tril=self.scale_tril,
            )
        elif get_backend() == ProbflowBackend.JAX:
            raise NotImplementedError("Wishart distribution is not yet implemented for JAX backend.")

        else:
            raise NotImplementedError("Wishart distribution is not yet implemented for TensorFlow backend.")

