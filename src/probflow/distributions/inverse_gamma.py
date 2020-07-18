from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class InverseGamma(BaseDistribution):
    r"""The Inverse-gamma distribution.

    The
    `Inverse-gamma distribution <https://en.wikipedia.org/wiki/Inverse-gamma_distribution>`_
    is a continuous distribution defined over all positive real numbers, and
    has two parameters:

    - a shape parameter (``shape`` or :math:`\alpha > 0`, a.k.a.
      "concentration"), and
    - a rate parameter (``rate`` or :math:`\beta > 0`, a.k.a. "scale").

    The ratio of :math:`\frac{\beta}{\alpha-1}` determines the mean of the
    distribution, and for :math:`\alpha > 2`, the variance is determined by:

    .. math ::

        \frac{\beta^2}{(\alpha-1)^2(\alpha-2)}

    A random variable :math:`x` drawn from an Inverse-gamma distribution

    .. math::

        x \sim \text{InvGamma}(\alpha, \beta)

    has probability

    .. math::

        p(x) = \frac{\beta^\alpha}{\Gamma (\alpha)} x^{-\alpha-1}
               \exp (-\frac{\beta}{x})

    Where :math:`\Gamma` is the
    `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.

    TODO: example image of the distribution


    Parameters
    ----------
    concentration : int, float, |ndarray|, or Tensor
        Shape parameter of the inverse gamma distribution (:math:`\alpha`).
    scale : int, float, |ndarray|, or Tensor
        Rate parameter of the inverse gamma distribution (:math:`\beta`).

    """

    def __init__(self, concentration, scale):

        # Check input
        ensure_tensor_like(concentration, "concentration")
        ensure_tensor_like(scale, "scale")

        # Store args
        self.concentration = concentration
        self.scale = scale

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch
            import torch.distributions as tod

            return tod.transformed_distribution.TransformedDistribution(
                tod.gamma.Gamma(self["concentration"], self["scale"]),
                tod.transforms.PowerTransform(torch.tensor([-1.0])),
            )
            # TODO: mean isn't implemented
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.InverseGamma(self["concentration"], self["scale"])
