from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class Gamma(BaseDistribution):
    r"""The Gamma distribution.

    The
    `Gamma distribution <https://en.wikipedia.org/wiki/Gamma_distribution>`_
    is a continuous distribution defined over all positive real numbers, and
    has two parameters:

    - a shape parameter (``shape`` or :math:`\alpha > 0`, a.k.a.
      "concentration"), and
    - a rate parameter (``rate`` or :math:`\beta > 0`).

    The ratio of :math:`\frac{\alpha}{\beta}` determines the mean of the
    distribution, and the ratio of :math:`\frac{\alpha}{\beta^2}` determines
    the variance.

    A random variable :math:`x` drawn from a Gamma distribution

    .. math::

        x \sim \text{Gamma}(\alpha, \beta)

    has probability

    .. math::

        p(x) = \frac{\beta^\alpha}{\Gamma (\alpha)} x^{\alpha-1}
               \exp (-\beta x)

    Where :math:`\Gamma` is the
    `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.

    TODO: example image of the distribution


    Parameters
    ----------
    shape : int, float, |ndarray|, or Tensor
        Shape parameter of the gamma distribution (:math:`\alpha`).
    rate : int, float, |ndarray|, or Tensor
        Rate parameter of the gamma distribution (:math:`\beta`).

    """

    def __init__(self, concentration, rate):

        # Check input
        ensure_tensor_like(concentration, "concentration")
        ensure_tensor_like(rate, "rate")

        # Store args
        self.concentration = concentration
        self.rate = rate

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.gamma.Gamma(self["concentration"], self["rate"])
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Gamma(self["concentration"], self["rate"])
