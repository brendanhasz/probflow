from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.torch_distributions import get_TorchDeterministic
from probflow.utils.validation import ensure_tensor_like


class Deterministic(BaseDistribution):
    r"""A deterministic distribution.

    A
    `deterministic distribution <https://en.wikipedia.org/wiki/Degenerate_distribution>`_
    is a continuous distribution defined over all real numbers, and has one
    parameter:

    - a location parameter (``loc`` or :math:`k_0`) which determines the mean
      of the distribution.

    A random variable :math:`x` drawn from a deterministic distribution
    has probability of 1 at its location parameter value, and zero elsewhere:

    .. math::

        p(x) =
        \begin{cases}
            1, & \text{if}~x=k_0 \\
            0, & \text{otherwise}
        \end{cases}

    TODO: example image of the distribution


    Parameters
    ----------
    loc : int, float, |ndarray|, or Tensor
        Mean of the deterministic distribution (:math:`k_0`).
        Default = 0
    """

    def __init__(self, loc=0):

        # Check input
        ensure_tensor_like(loc, "loc")

        # Store args
        self.loc = loc

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            TorchDeterministic = get_TorchDeterministic()
            return TorchDeterministic(self["loc"])
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Deterministic(self["loc"])
