"""Probability distributions.

TODO: more info

Continuous Distributions
------------------------

* :class:`.Normal`
* :class:`.HalfNormal`
* :class:`.StudentT`
* :class:`.Cauchy`
* :class:`.Gamma`
* :class:`.InvGamma`

Discrete Distributions
----------------------

* :class:`.Bernoulli`
* :class:`.Poisson`

TODO: Categorical, etc

----------

"""

__all__ = [
    'Normal',
    'HalfNormal',
    'StudentT',
    'Cauchy',
    'Gamma',
    'InvGamma',
    'Bernoulli',
    'Poisson',
]

from collections import OrderedDict

import tensorflow_probability as tfp
tfd = tfp.distributions

from .core import ContinuousDistribution, DiscreteDistribution, REQUIRED



class Normal(ContinuousDistribution):
    r"""The Normal distribution.

    TODO: more... This is an :math:`\alpha=3` inline alpha

    .. math::

        y \sim \mathcal{N}(0, 1)

    .. math::

        p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}}
               \exp \left( -\frac{(x-\mu)^2}{2 \sigma^2} \right)

    non-inline math

    Parameters
    ----------
    loc : int, float, |ndarray|, |Tensor|, |Parameter|, or |Layer|
        Mean of the normal distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, |Tensor|, |Parameter|, or |Layer|
        Standard deviation of the normal distribution (:math:`\sigma^2`).
        Default = 1

    """


    # Distribution parameters and their default values
    _default_args = OrderedDict([
        ('loc', 0),
        ('scale', 1)
    ])

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'loc': (None, None),
        'scale': (0, None)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.Normal(loc=args['loc'], scale=args['scale'])



class HalfNormal(ContinuousDistribution):
    r"""The Half-normal distribution.


    TODO: More info...

    .. math::

        p(x) =
        \begin{cases}
            0, & \text{if}~x<0 \\
            \frac{2}{\sqrt{2 \pi \sigma^2}}
            \exp \left( -\frac{(x-\mu)^2}{2 \sigma^2} \right),
            & \text{otherwise}
        \end{cases}



    """


    # Distribution parameter and the default value
    _default_args = {
        'scale': 1
    }

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'scale': (0, None)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.HalfNormal(scale=args['scale'])



class StudentT(ContinuousDistribution):
    r"""The Student-t distribution.


    TODO: More info...


    """


    # Distribution parameters and their default values
    _default_args = OrderedDict([
        ('df', 1),
        ('loc', 0),
        ('scale', 1)
    ])

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'df': (0, None),
        'loc': (None, None),
        'scale': (0, None)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.StudentT(args['df'], args['loc'], args['scale'])



class Cauchy(ContinuousDistribution):
    r"""The Cauchy distribution.


    TODO: More info...


    """


    # Distribution parameters and their default values
    _default_args = OrderedDict([
        ('loc', 0),
        ('scale', 1)
    ])

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'loc': (None, None),
        'scale': (0, None)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.Cauchy(args['loc'], args['scale'])



class Gamma(ContinuousDistribution):
    r"""The Gamma distribution.

    TODO: more...

    .. math::

        y \sim \text{Gamma}(\alpha, \beta)

    .. math::

        p(x) = \frac{\beta^\alpha}{\Gamma (\alpha)} x^{\alpha-1}
               \exp (-\beta x)


    Parameters
    ----------
    shape : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Shape parameter of the gamma distribution (:math:`\alpha`).
    rate : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Rate parameter of the gamma distribution (:math:`\beta`).

    """


    # Distribution parameters and their default values
    _default_args = OrderedDict([
        ('shape', REQUIRED),
        ('rate', REQUIRED)
    ])

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'shape': (0, None),
        'rate': (0, None)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.Gamma(concentration=args['shape'], rate=args['rate'])



class InvGamma(ContinuousDistribution):
    r"""The Inverse-gamma distribution.

    TODO: more...

    .. math::

        y \sim \text{InvGamma}(\alpha, \beta)

    .. math::

        p(x) = \frac{\beta^\alpha}{\Gamma (\alpha)} x^{-\alpha-1}
               \exp (-\frac{\beta}{x})


    Parameters
    ----------
    shape : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Shape parameter of the inverse gamma distribution (:math:`\alpha`).
    rate : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Rate parameter of the inverse gamma distribution (:math:`\beta`).

    """


    # Distribution parameters and their default values
    _default_args = OrderedDict([
        ('shape', REQUIRED),
        ('rate', REQUIRED)
    ])

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'shape': (0, None),
        'rate': (0, None)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.InverseGamma(concentration=args['shape'], rate=args['rate'])



class Bernoulli(DiscreteDistribution):
    r"""The Bernoulli distribution.


    TODO: More info...


    """

    # Default kwargs
    _default_kwargs = {
        'input_type': 'logits'
    }

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'input': (None, None)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        if self.kwargs['input_type'] == 'logits': #input arg is the logits
            return tfd.Bernoulli(logits=args['input'])
        elif self.kwargs['input_type'] == 'probs': #input arg is the raw probs
            return tfd.Bernoulli(logits=args['input'])
        else:
            raise TypeError('Bernoulli kwarg input_type must be either ' +
                            '\'logits\' or \'probs\'')



class Poisson(DiscreteDistribution):
    r"""The Poisson distribution.


    TODO: More info...


    """


    # Distribution parameter and the default value
    _default_args = {
        'rate': REQUIRED
    }

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'rate': (0, None)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.Poisson(args['rate'])



# TODO: will have to be some way to distinguish batch_size from dimensions from
# number of independent dists?
# e.g. a MultivariateNormal dist w/ shape (3, 4, 5).
# Is that batch_size=3, dimensions=4, and 5 independent dists?
# or should the last 2 be flipped?  TFP uses batch_shape and event_shape
# will have the same problem w/ any multidim dist, e.g. Dirichlet, Multinomial

# TODO: other common distributions, esp Categorical, Binomial
# and really there's Discrete models but then there's Categorical models...
# ie you can get the cum prob value on a poisson but not on a categorical...

# also at some point:
# MultivariateNormal, mvt, mvc, Exponential, Beta, Gamma
# Binomial, BetaBinomial
