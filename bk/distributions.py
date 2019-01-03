"""Probability distributions.

TODO: more info...

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
    """TODO


    TODO: More info...


    """


    # Distribution parameters and their default values
    default_args = {
        'loc': 0,
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.Normal(loc=args['loc'], scale=args['scale'])



class HalfNormal(ContinuousDistribution):
    """TODO


    TODO: More info...


    """


    # Distribution parameters and their default values
    default_args = {
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.HalfNormal(scale=args['scale'])



class StudentT(ContinuousDistribution):
    """TODO


    TODO: More info...


    """


    # Distribution parameters and their default values
    default_args = {
        'df': 1,
        'loc': 0,
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.StudentT(args['df'], args['loc'], args['scale'])



class Cauchy(ContinuousDistribution):
    """TODO


    TODO: More info...


    """


    # Distribution parameters and their default values
    default_args = {
        'loc': 0,
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.Cauchy(args['loc'], args['scale'])


# TODO: other common distributions, esp Bernoulli and Poisson
