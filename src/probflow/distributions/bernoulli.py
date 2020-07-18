from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class Bernoulli(BaseDistribution):
    r"""The Bernoulli distribution.

    The
    `Bernoulli distribution <https://en.wikipedia.org/wiki/Bernoulli_distribution>`_
    is a discrete distribution defined over only two integers: 0 and 1.
    It has one parameter:

    - a probability parameter (:math:`0 \leq p \leq 1`).

    A random variable :math:`x` drawn from a Bernoulli distribution

    .. math::

        x \sim \text{Bernoulli}(p)

    takes the value :math:`1` with probability :math:`p`, and takes the value
    :math:`0` with probability :math:`p-1`.

    TODO: example image of the distribution

    TODO: specifying either logits or probs


    Parameters
    ----------
    logits : int, float, |ndarray|, or Tensor
        Logit-transformed probability parameter of the  Bernoulli
        distribution (:math:`\p`)
    probs : int, float, |ndarray|, or Tensor
        Logit-transformed probability parameter of the  Bernoulli
        distribution (:math:`\p`)
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

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.bernoulli.Bernoulli(
                logits=self["logits"], probs=self["probs"]
            )
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Bernoulli(logits=self["logits"], probs=self["probs"])
