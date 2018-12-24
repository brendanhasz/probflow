"""Transformations.

TODO: more info...
if input is tensor, they should return a (transformed) tensor
if input is distribution, return a bijected distribution

So, transformations can be applied to anything and 
if you pass them a Model offspring, they'll return a model
but so we want to be able to call .fit() and stuff on them...?
sigh these should really just be a func, actually

"""



import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from layers import BaseLayer
from base_models import BaseModel
from variables import Variable


class BaseTransformation(ABC):
    """Abstract transformation class (just used as an implementation base)


    TODO: More info...


    """

    # TODO

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Construct layer.

        Inheriting class must define how to initialize the layer.
        """
        pass


    @abstractmethod
    def build(self, data):
        """Build the tensorflow graph for this layer and its arguments.

        Inheriting class must define how to build the layer.  Should return a 
        tuple where the first element is the built model and the second element
        is the mean model (model evaluated with the mean of each variables 
        variational distribution).
        """
        pass       



class Exp(BaseTransformation):
    """Transform the input `x` by `exp(x)`. 


    TODO: More info...


    """

    def __init__(self, input):
        """Construct transformation.

        TODO: docs...
        """

        # Check input is of correct type
        # TODO

        # Assign attribute
        self.input = input


    def build(self, data):
        """Build the tensorflow graph for this transformation and its arguments.

        TODO: docs...
        """
        if isinstance(self.input, (int, float, np.ndarray)):
            self.built_trans = np.exp(self.input)
            self.mean_trans = np.exp(self.input)
        elif isinstance(self.input, (tf.Tensor, BaseLayer)):
            self.built_trans = tf.math.exp(self.input)
            self.built_trans = tf.math.exp(self.input)
        elif isinstance(self.input, BaseModel):
            # TODO: build input (will return a tfp.distribution)
            # TODO: use bijector
        elif isinstance(self.input, BaseTransformation):
            # TODO: build input
            # TODO: use bijector
        else:
            pass # TypeError should have been raised by __init__



class Log(BaseTransformation):
    """Transform the input `x` by `log(x)`. 


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
            