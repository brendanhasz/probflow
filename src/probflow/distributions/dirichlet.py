from probflow.utils.base import BaseDistribution
from probflow.utils.settings import get_backend
from probflow.utils.validation import ensure_tensor_like


class Dirichlet(BaseDistribution):
    r"""The Dirichlet distribution.

    The
    `Dirichlet distribution <http://en.wikipedia.org/wiki/Dirichlet_distribution>`_
    is a continuous distribution defined over the :math:`k`-simplex, and has
    one vector of parameters:

    - concentration parameters (``concentration`` or
      :math:`\boldsymbol{\alpha} \in \mathbb{R}^{k}_{>0}`),
      a vector of positive numbers which determine the relative likelihoods of
      different categories represented by the distribution.

    A random variable (a vector) :math:`\mathbf{x}` drawn from a Dirichlet
    distribution

    .. math::

        \mathbf{x} \sim \text{Dirichlet}(\boldsymbol{\alpha})

    has probability

    .. math::

        p(\mathbf{x}) = \frac{1}{\mathbf{\text{B}}(\boldsymbol{\alpha})}
                        \prod_{i=1}^K x_i^{\alpha_i-1}

    where :math:`\mathbf{\text{B}}` is the multivariate beta function.

    TODO: example image of the distribution


    Parameters
    ----------
    concentration : |ndarray|, or Tensor
        Concentration parameter of the Dirichlet distribution (:math:`\alpha`).
    """

    def __init__(self, concentration):

        # Check input
        ensure_tensor_like(concentration, "concentration")

        # Store args
        self.concentration = concentration

    def __call__(self):
        """Get the distribution object from the backend"""
        if get_backend() == "pytorch":
            import torch.distributions as tod

            return tod.dirichlet.Dirichlet(self["concentration"])
        else:
            from tensorflow_probability import distributions as tfd

            return tfd.Dirichlet(self["concentration"])
