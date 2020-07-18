from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class StudentT(BaseDistribution):
    r"""The Student-t distribution.

    The
    `Student's t-distribution <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_
    is a continuous distribution defined over all real numbers, and has three
    parameters:

    - a degrees of freedom parameter (``df`` or :math:`\nu > 0`), which
      determines how many degrees of freedom the distribution has,
    - a location parameter (``loc`` or :math:`\mu`) which determines the mean
      of the distribution, and
    - a scale parameter (``scale`` or :math:`\sigma > 0`) which determines the
      standard deviation of the distribution.

    A random variable :math:`x` drawn from a Student's t-distribution

    .. math::

        x \sim \text{StudentT}(\nu, \mu, \sigma)

    has probability

    .. math::

        p(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu x} \Gamma
            (\frac{\nu}{2})} \left( 1 + \frac{x^2}{\nu} \right)^
            {-\frac{\nu+1}{2}}

    Where :math:`\Gamma` is the
    `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.

    TODO: example image of the distribution


    Parameters
    ----------
    df : int, float, |ndarray|, or Tensor
        Degrees of freedom of the t-distribution (:math:`\nu`).
        Default = 1
    loc : int, float, |ndarray|, or Tensor
        Median of the t-distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, or Tensor
        Spread of the t-distribution (:math:`\sigma`).
        Default = 1
    """

    def __init__(self, df=1, loc=0, scale=1):

        # Check input
        ensure_tensor_like(df, "df")
        ensure_tensor_like(loc, "loc")
        ensure_tensor_like(scale, "scale")

        # Store args
        self.df = df
        self.loc = loc
        self.scale = scale

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.studentT.StudentT(
                self["df"], self["loc"], self["scale"]
            )
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.StudentT(self["df"], self["loc"], self["scale"])

    def mean(self):
        """Compute the mean of this distribution.

        Note that the mean of a StudentT distribution is technically
        undefined when df=1.
        """
        return self.loc
