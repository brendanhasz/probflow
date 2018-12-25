"""Probability distributions.

TODO: more info...

"""



import tensorflow_probability as tfp
tfd = tfp.distributions


# TODO: define continuous vs categorical distributions? (and inherit accordingly?)

class ContinuousDistribution(ContinuousModel):
    """TODO


    TODO: More info...


    """
    

    def _log_loss(self, obj, vals):
        """Compute the log loss ."""
        return obj.log_prob(vals)



class Normal(ContinuousDistribution):
    """TODO


    TODO: More info...


    """


    # Distribution parameters and their default values
    self.default_args = {
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
    self.default_args = {
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model."""
        return tfd.HalfNormal(scale=args['scale'])



# TODO: other common distributions
