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
        """Build the distribution model.

        TODO: Docs...

        """
        return tfd.Normal(loc=args['loc'],
                          scale=args['scale'])


    def _log_loss(self, args, vals):
        """Compute the loss due to the value."""
        tfd.Normal(loc=args['loc'], scale=args['scale']).log_prob(vals)
        # TODO: so the values are the sampled values? so self.built_model.sample()?
        # but the parent calls that so how can we compute it...? 
        # may have to have parent call this in sum_arg_losses (when this obj is an arg)
        # then fit() calls this with vals = the y variable


class HalfNormal(ContinuousModel):
    """TODO


    TODO: More info...


    """


    # Distribution parameters and their default values
    self.default_args = {
        'scale': 1
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """
        return tfd.HalfNormal(scale=args['scale'])


# TODO: other common distributions
