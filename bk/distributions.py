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

TODO: Bernoulli, Categorical, etc

----------

"""



import tensorflow_probability as tfp
tfd = tfp.distributions

from .core import ContinuousModel, CategoricalModel



class BaseDistribution():
    """

    TODO: More info...

    """

    def _log_loss(self, obj, vals):
        """Compute the log loss ."""
        return tf.reduce_mean(obj.log_prob(vals))



class ContinuousDistribution(BaseDistribution, ContinuousModel):
    """TODO


    TODO: More info...


    """
    pass



class CategoricalDistribution(BaseDistribution, CategoricalModel):
    """TODO


    TODO: More info...


    """
    pass



class Normal(ContinuousDistribution):
    r"""The Normal distribution.

    TODO: more... This is an :math:`\alpha=3` inline alpha

    .. math::

        y \sim \mathcal{N}(0, 1)

    .. math::

        p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} 
               \exp (-\frac{(x-\mu)^2}{2 \sigma^2})

    non-inline math

    Parameters
    ----------
    loc : int, float, np.ndarray, or tf.Tensor, optional
        Mean of the normal distribution (:math:`\mu`). 
        Default = 0
    scale : int, float, np.ndarray, or tf.Tensor, optional
        Standard deviation of the normal distribution (:math:`\sigma^2`). 
        Default = 1

    """


    # Distribution parameters and their default values
    _default_args = {
        'loc': 0,
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.Normal(loc=args['loc'], scale=args['scale'])



class HalfNormal(ContinuousDistribution):
    r"""The Half-normal distribution.


    TODO: More info...


    """


    # Distribution parameters and their default values
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
    _default_args = {
        'df': 1,
        'loc': 0,
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.StudentT(args['df'], args['loc'], args['scale'])



class Cauchy(ContinuousDistribution):
    r"""The Cauchy distribution.


    TODO: More info...


    """


    # Distribution parameters and their default values
    _default_args = {
        'loc': 0,
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.Cauchy(args['loc'], args['scale'])


# TODO: other common distributions, esp Bernoulli and Poisson
