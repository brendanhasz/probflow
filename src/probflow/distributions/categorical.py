import probflow.utils.ops as O
from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
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

    def __init__(self, logits=None, probs=None):

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
            self.ndim = len(probs.shape)
        else:
            self.ndim = len(logits.shape)

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.categorical.Categorical(
                logits=self["logits"], probs=self["probs"]
            )
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Categorical(logits=self["logits"], probs=self["probs"])

    def prob(self, y):
        """Doesn't broadcast correctly when logits/probs and y are same dims"""
        if self.ndim == len(y.shape):
            y = O.squeeze(y)
        return super().prob(y)

    def log_prob(self, y):
        """Doesn't broadcast correctly when logits/probs and y are same dims"""
        if self.ndim == len(y.shape):
            y = O.squeeze(y)
        return super().log_prob(y)
