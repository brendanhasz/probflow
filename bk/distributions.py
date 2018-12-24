"""Probability distributions.

TODO: more info...

"""



import tensorflow_probability as tfp
tfd = tfp.distributions


# TODO: define continuous vs categorical distributions? (and inherit accordingly?)


class Normal(ContinuousModel):
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


    def _log_loss(self, obj, vals):
        """Compute the loss due to the value."""
        return obj.log_prob(vals)
        # TODO: well, this returns *a* normal distribution, but is that what we want?
        # don't we really want to do self.built_model.log_prob(vals)?
        # but that will only work if it's a distribution and instead of args we pass model
        # will it work for layers?



class HalfNormal(ContinuousModel):
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


    def _log_loss(self, args, vals):
        """Compute the loss due to the value."""
        # TODO: same as 



# TODO: other common distributions
