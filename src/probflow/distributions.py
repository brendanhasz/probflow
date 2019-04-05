"""Probability distributions.

TODO: more info

Continuous Distributions
------------------------

* :class:`.Deterministic`
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
    'Deterministic',
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

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .core import ContinuousDistribution, DiscreteDistribution, REQUIRED



class Deterministic(ContinuousDistribution):
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
    loc : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Mean of the deterministic distribution (:math:`k_0`).
        Default = 0
    """

    # Distribution parameters and their default values
    _default_args = OrderedDict([
        ('loc', 0),
    ])

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'loc': (None, None),
    }

    # Posterior parameter initializers
    _post_param_init = {
        'loc': tf.initializers.truncated_normal(mean=0.0, stddev=1.0),
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.Deterministic(loc=args['loc'])



class Normal(ContinuousDistribution):
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
    loc : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Mean of the normal distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Standard deviation of the normal distribution (:math:`\sigma`).
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

    # Posterior parameter initializers
    _post_param_init = {
        'loc': tf.initializers.truncated_normal(mean=0.0, stddev=1.0),
        'scale': tf.initializers.random_uniform(minval=-0.7, maxval=0.4)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.Normal(loc=args['loc'], scale=args['scale'])



class HalfNormal(ContinuousDistribution):
    r"""The Half-normal distribution.

    The half-normal distribution is a continuous distribution defined over all
    positive real numbers, and has one parameter:

    - a scale parameter (``scale`` or :math:`\sigma > 0`) which determines the
      standard deviation of the distribution.

    A random variable :math:`x` drawn from a half-normal distribution

    .. math::

        x \sim \text{HalfNormal}(\sigma)

    has probability

    .. math::

        p(x) =
        \begin{cases}
            0, & \text{if}~x<0 \\
            \frac{2}{\sqrt{2 \pi \sigma^2}}
            \exp \left( -\frac{(x-\mu)^2}{2 \sigma^2} \right),
            & \text{otherwise}
        \end{cases}

    TODO: example image of the distribution


    Parameters
    ----------
    scale : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Standard deviation of the underlying normal distribution 
        (:math:`\sigma`).
        Default = 1
    """

    # Distribution parameter and the default value
    _default_args = {
        'scale': 1
    }

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'scale': (0, None)
    }

    # Posterior parameter initializers
    _post_param_init = {
        'scale': tf.initializers.random_uniform(minval=-0.7, maxval=0.4)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.HalfNormal(scale=args['scale'])



class StudentT(ContinuousDistribution):
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
    df : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Degrees of freedom of the t-distribution (:math:`\nu`).
        Default = 1
    loc : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Median of the t-distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Spread of the t-distribution (:math:`\sigma`).
        Default = 1
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

    # Posterior parameter initializers
    _post_param_init = {
        'df': tf.keras.initializers.Constant(value=1),
        'loc': tf.initializers.truncated_normal(mean=0.0, stddev=1.0),
        'scale': tf.initializers.random_uniform(minval=-0.7, maxval=0.4)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.StudentT(args['df'], args['loc'], args['scale'])



class Cauchy(ContinuousDistribution):
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
    loc : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Median of the Cauchy distribution (:math:`\mu`).
        Default = 0
    scale : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Spread of the Cauchy distribution (:math:`\gamma`).
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

    # Posterior parameter initializers
    _post_param_init = {
        'loc': tf.initializers.truncated_normal(mean=0.0, stddev=1.0),
        'scale': tf.initializers.random_uniform(minval=-0.7, maxval=0.4)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.Cauchy(args['loc'], args['scale'])



class Gamma(ContinuousDistribution):
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

    # Posterior parameter initializers
    _post_param_init = {
        'shape': tf.initializers.truncated_normal(mean=1.6, stddev=0.1),
        'rate': tf.initializers.truncated_normal(mean=1.6, stddev=0.1)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.Gamma(concentration=args['shape'], rate=args['rate'])



class InvGamma(ContinuousDistribution):
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

    # Posterior parameter initializers
    _post_param_init = {
        'shape': tf.initializers.truncated_normal(mean=1.6, stddev=0.1),
        'rate': tf.initializers.truncated_normal(mean=1.6, stddev=0.1)
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        return tfd.InverseGamma(concentration=args['shape'], rate=args['rate'])



class Bernoulli(DiscreteDistribution):
    r"""The Bernoulli distribution.

    The 
    `Bernoulli distribution <https://en.wikipedia.org/wiki/Bernoulli_distribution>`_
    is a discrete distribution defined over only two integers: 0 and 1.
    It has one parameter: 

    - a probability parameter (:math:`0 \leq p \leq 1`).

    The ratio of :math:`\frac{\beta}{\alpha-1}` determines the mean of the
    distribution, and for :math:`\alpha > 2`, the variance is determined by:

    .. math ::

        \frac{\beta^2}{(\alpha-1)^2(\alpha-2)}

    A random variable :math:`x` drawn from a Bernoulli distribution

    .. math::

        x \sim \text{Bernoulli}(p)

    takes the value :math`1` with probability :math:`p`, and takes the value
    :math:`0` with probability :math:`p-1`.

    TODO: example image of the distribution


    Parameters
    ----------
    p : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Probability parameter of the Bernoulli distribution (:math:`\p`).
    input_type : str ('logits' or 'probs')
        How to interperet the probability parameter ``p``.  If ``'probs'``,
        ``p`` represents the raw probability.  If ``'logit'``, ``p`` 
        represents the logit-transformed probability.
    """

    # Default kwargs
    _default_kwargs = {
        'input_type': 'logits'
    }

    # Distribution parameters and their default values
    _default_args = OrderedDict([
        ('p', REQUIRED),
    ])

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'p': (None, None)
    }

    # Posterior parameter initializers
    _post_param_init = {
        'p': tf.initializers.truncated_normal(mean=0.0, stddev=1.0),
    }

    def _build(self, args, _data, _batch_shape):
        """Build the distribution model."""
        if self.kwargs['input_type'] == 'logits': #p arg is the logits
            return tfd.Bernoulli(logits=args['p'])
        elif self.kwargs['input_type'] == 'probs': #p arg is the raw probs
            return tfd.Bernoulli(logits=args['p'])
        else:
            raise TypeError('Bernoulli kwarg input_type must be either ' +
                            '\'logits\' or \'probs\'')



class Poisson(DiscreteDistribution):
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
    rate : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Rate parameter of the Poisson distribution (:math:`\lambda`).
    """

    # Distribution parameter and the default value
    _default_args = {
        'rate': REQUIRED
    }

    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'rate': (0, None)
    }

    # Posterior parameter initializers
    _post_param_init = {
        'rate': tf.initializers.random_uniform(minval=0.0, maxval=3.0),
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
