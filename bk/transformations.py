"""Transformations.

TODO: more info...
if input is tensor, they should return a (transformed) tensor
if input is distribution, return a bijected distribution

"""



import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class Exp(ContinuousModel):
    """TODO


    TODO: More info...


    """

    # Transformation parameters and their default values
    self.default_args = {
        'val': None
    }
    

    def _build(self, args, data):
        """Build the transformation model.

        TODO: Docs...

        """
        if self._arg_is('number', 'val'):
            return np.exp(args['val'])
        elif self._arg_is('tensor', 'val'):
            return tf.math.exp(args['val'])



class Log(ContinuousModel):
    """TODO


    TODO: More info...


    """

    # Transformation parameters and their default values
    self.default_args = {
        'val': None
    }
    

    def _build(self, args, data):
        """Build the transformation model.

        TODO: Docs...

        """
        if self._arg_is('number', 'val'):
            return np.log(args['val'])
        elif self._arg_is('tensor', 'val'):
            return tf.math.log(args['val'])
            