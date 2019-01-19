"""Probability distributions.

TODO: more info

TODO: list the distributions by type (continuous, categorical, etc), linking to api refs below

Continuous Distributions
------------------------

* :class:`.Normal`
* :class:`.HalfNormal`
* :class:`.StudentT`
* :class:`.Cauchy`

Discrete Distributions
----------------------

* :class:`.Bernoulli`
* :class:`.Poisson`

TODO: Categorical, etc

----------

"""



import tensorflow_probability as tfp
tfd = tfp.distributions

from .core import ContinuousDistribution, DiscreteDistribution



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
    loc : int, float, |ndarray|, |Tensor|, |Variable|, or |Layer|
        Mean of the normal distribution (:math:`\mu`). 
        Default = 0
    scale : int, float, |ndarray|, |Tensor|, |Variable|, or |Layer|
        Standard deviation of the normal distribution (:math:`\sigma^2`). 
        Default = 1

    """


    # Distribution parameters and their default values
    _default_args = OrderedDict([
        ('loc', 0),
        ('scale', 1)
    ])    


    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.Normal(loc=args['loc'], scale=args['scale'])



class HalfNormal(ContinuousDistribution):
    r"""The Half-normal distribution.


    TODO: More info...


    """


    # Distribution parameter and the default value
    _default_args = {
        'scale': 1
    }
    

    def _build(self, args, data):
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
    

    def _build(self, args, data):
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
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.Cauchy(args['loc'], args['scale'])



class Bernoulli(DiscreteDistribution):
    r"""The Bernoulli distribution.


    TODO: More info...


    """


    # Default kwargs
    _default_kwargs = {
        'input_type': 'logits'
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        if self.kwargs['input_type']=='logits': #input arg is the logits
            return tfd.Bernoulli(logits=args['input'])
        elif self.kwargs['input_type']=='probs': #input arg is the raw probs
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
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.Poisson(args['rate'])



# TODO: will have to be some way to distinguish batch_size from dimensions from number of independent dists?
# e.g. a MultivariateNormal dist w/ shape (3, 4, 5). Is that batch_size=3, dimensions=4, and 5 independent dists? 
# or should the last 2 be flipped?  TFP handles this w/ batch_shape and event_shape
# will have the same problem w/ any multidim prob dist, e.g. Dirichlet, Multinomial

# TODO: other common distributions, esp Categorical, Binomial
# and really there's Discrete models but then there's Categorical models...
# ie you can get the cum prob value on a poisson but not on a categorical...

# also at some point: 
# MultivariateNormal, mvt, mvc, Exponential, Beta, Gamma
# Binomial, BetaBinomial