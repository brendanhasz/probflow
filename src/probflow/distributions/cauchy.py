from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class Cauchy(BaseDistribution):
    r"""The Cauchy distribution.

    The
    `Cauchy distribution <https://en.wikipedia.org/wiki/Cauchy_distribution>`_
    is a continuous distribution defined over all real numbers, and has two
    parameters:

    - a location parameter (``loc`` or :math:`\mu`) which determines the
      median of the distribution, and
    - a scale parameter (``scale`` or :math:`\gamma > 0`) which determines the
      spread of the distribution.

    A random variable :math:`x` drawn from a Cauchy distribution

    .. math::

        x \sim \text{Cauchy}(\mu, \gamma)

    has probability

    .. math::

        p(x) = \frac{1}{\pi \gamma \left[  1 +
               \left(  \frac{x-\mu}{\gamma} \right)^2 \right]}

    The Cauchy distribution is equivalent to a Student's t-distribution with
    one degree of freedom.

    TODO: example image of the distribution


    Parameters
    ----------
    loc : int, float, |ndarray|, or Tensor
        Median of the Cauchy distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, or Tensor
        Spread of the Cauchy distribution (:math:`\gamma`).
        Default = 1
    """

    def __init__(self, loc=0, scale=1):

        # Check input
        ensure_tensor_like(loc, "loc")
        ensure_tensor_like(scale, "scale")

        # Store args
        self.loc = loc
        self.scale = scale

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.cauchy.Cauchy(self["loc"], self["scale"])
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Cauchy(self["loc"], self["scale"])

    def mean(self):
        """Compute the mean of this distribution.

        Note that the mean of a Cauchy distribution is technically undefined.
        """
        return self.loc
