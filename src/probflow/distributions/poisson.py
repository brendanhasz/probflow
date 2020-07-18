from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class Poisson(BaseDistribution):
    r"""The Poisson distribution.

    The
    `Poisson distribution <https://en.wikipedia.org/wiki/Poisson_distribution>`_
    is a discrete distribution defined over all non-negativve real integers,
    and has one parameter:

    - a rate parameter (``rate`` or :math:`\lambda`) which determines the mean
      of the distribution.

    A random variable :math:`x` drawn from a Poisson distribution

    .. math::

        x \sim \text{Poisson}(\lambda)

    has probability

    .. math::

        p(x) = \frac{\lambda^x e^{-\lambda}}{x!}

    TODO: example image of the distribution


    Parameters
    ----------
    rate : int, float, |ndarray|, or Tensor
        Rate parameter of the Poisson distribution (:math:`\lambda`).
    """

    def __init__(self, rate):

        # Check input
        ensure_tensor_like(rate, "rate")

        # Store args
        self.rate = rate

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.poisson.Poisson(self["rate"])
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Poisson(self["rate"])
