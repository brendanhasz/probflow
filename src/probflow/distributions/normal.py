from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class Normal(BaseDistribution):
    r"""The Normal distribution.

    The
    `normal distribution <https://en.wikipedia.org/wiki/Normal_distribution>`_
    is a continuous distribution defined over all real numbers, and has two
    parameters:

    - a location parameter (``loc`` or :math:`\mu`) which determines the mean
      of the distribution, and
    - a scale parameter (``scale`` or :math:`\sigma > 0`) which determines the
      standard deviation of the distribution.

    A random variable :math:`x` drawn from a normal distribution

    .. math::

        x \sim \mathcal{N}(\mu, \sigma)

    has probability

    .. math::

        p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}}
               \exp \left( -\frac{(x-\mu)^2}{2 \sigma^2} \right)

    TODO: example image of the distribution


    Parameters
    ----------
    loc : int, float, |ndarray|, or Tensor
        Mean of the normal distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, or Tensor
        Standard deviation of the normal distribution (:math:`\sigma`).
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

            return tod.normal.Normal(self["loc"], self["scale"])
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Normal(self["loc"], self["scale"])
