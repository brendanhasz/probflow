"""Probability distributions.

TODO: more info...

"""



import tensorflow_probability as tfp
tfd = tfp.distributions



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
