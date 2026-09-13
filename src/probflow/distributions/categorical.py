"""The Categorical distribution."""

import probflow.utils.ops as O
from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.shape import get_ndims
from probflow.utils.typing import BackendDistribution, TensorLike
from probflow.utils.validation import ensure_tensor_like


class Categorical(BaseDistribution):
    r"""The Categorical distribution.

    The
    `Categorical distribution <https://en.wikipedia.org/wiki/Categorical_distribution>`_
    is a discrete distribution defined over :math:`N` integers: 0 through
    :math:`N-1`. A random variable :math:`x` drawn from a Categorical
    distribution

    .. math::

        x \sim \text{Categorical}(\mathbf{\theta})

    has probability

    .. math::

        p(x=i) = p_i

    TODO: example image of the distribution

    TODO: logits vs probs


    Parameters
    ----------
    logits : int, float, |ndarray|, or Tensor
        Logit-transformed category probabilities
        (:math:`\frac{\mathbf{\theta}}{1-\mathbf{\theta}}`)
    probs : int, float, |ndarray|, or Tensor
        Raw category probabilities (:math:`\mathbf{\theta}`)
    """

    def __init__(
        self, logits: TensorLike | None = None, probs: TensorLike | None = None
    ) -> None:

        # Check input
        if logits is None and probs is None:
            raise TypeError("either logits or probs must be specified")
        if logits is None:
            ensure_tensor_like(probs, "probs")
        if probs is None:
            ensure_tensor_like(logits, "logits")

        # Store args
        self.logits = logits
        self.probs = probs
        if logits is None:
            self.ndim = get_ndims(probs, "probs")
        else:
            self.ndim = get_ndims(logits, "logits")

    def __call__(self) -> BackendDistribution:
        """Get the distribution object from the backend."""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.categorical.Categorical(
                logits=self["logits"], probs=self["probs"]
            )
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Categorical(logits=self["logits"], probs=self["probs"])

    def prob(self, y: TensorLike) -> TensorLike:
        """Doesn't broadcast correctly when logits/probs and y are same dims."""
        if self.ndim == get_ndims(y):
            y = O.squeeze(y)
        return super().prob(y)

    def log_prob(self, y: TensorLike) -> TensorLike:
        """Doesn't broadcast correctly when logits/probs and y are same dims."""
        if self.ndim == get_ndims(y):
            y = O.squeeze(y)
        return super().log_prob(y)

    def mean(self) -> TensorLike:
        """Since this is a categorical distribution, return the mode."""
        # PyTorch mean method returns the mode
        if get_backend() == "pytorch":
            return super().mode()
        # But TensorFlow returns a float value for the mean, so need to use mode instead
        else:
            return super().mode()
