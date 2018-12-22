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
    

    def build(self, data):
        """Build the distribution model.

        TODO: Docs...

        """
        self.build_args(data)
        return tfd.Normal(loc=self.built_args['loc'],
                          scale=self.built_args['scale'])



class HalfNormal(ContinuousModel):
    """TODO


    TODO: More info...


    """


    # Distribution parameters and their default values
    self.default_args = {
        'scale': 1
    }
    

    def build(self, data):
        """Build the distribution model.

        TODO: Docs...

        """
        self.build_args(data)
        return tfd.HalfNormal(scale=self.built_args['scale'])
