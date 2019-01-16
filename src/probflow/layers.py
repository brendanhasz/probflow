"""Layers.

TODO: more info...
A layer, unlike a model, returns a single value
whereas a model returns a probability distribution!


Data Layers
-----------

* :class:`.Input`

Basic Arithmetic Layers
-----------------------

* :class:`.Add`
* :class:`.Sub`
* :class:`.Mul`
* :class:`.Div`
* :class:`.Neg`
* :class:`.Abs`
* :class:`.Exp`
* :class:`.Log`

Neural Network Layers
---------------------

* :class:`.Dense`
* :class:`.Sequential`
* :class:`.Embedding`

----------

"""

from collections import OrderedDict
import numpy as np
import tensorflow as tf

from .distributions import Normal
from .core import BaseLayer, REQUIRED



class Input(BaseLayer):
    """Layer which represents the input data.


    TODO: More info...


    """


    # Input layer takes no parameters
    _default_args = dict()
 

    # Default kwargs
    _default_kwargs = {
        'cols': None #indexes of columns to use
    }
   

    def _build(self, args, data):
        """Build the layer."""
        if isinstance(args['cols'], np.ndarray) and len(args['cols'])==0:
            return data
        else:
            # TODO: slice data by cols
            pass



class Add(BaseLayer):
    """A layer which adds two inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])
    

    def _build(self, args, data):
        """Build the layer."""
        return args['a'] + args['b']



class Sub(BaseLayer):
    """A layer which subtracts one input from another.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])
    

    def _build(self, args, data):
        """Build the layer."""
        return args['a'] - args['b']



class Mul(BaseLayer):
    """A layer which multiplies two inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])
    

    def _build(self, args, data):
        """Build the layer."""
        # TODO: have to distinguish between matrix and elementwise mult somehow?
        return args['a'] * args['b']



class Div(BaseLayer):
    """A layer which divides one input by another.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])
    

    def _build(self, args, data):
        """Build the layer."""
        return args['a'] / args['b']



class Neg(BaseLayer):
    """A layer which outputs the negative of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return -args['input']



class Abs(BaseLayer):
    """A layer which outputs the absolute value of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return abs(args['input'])



class Exp(BaseLayer):
    """A layer which outputs the natural exponent of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.exp(args['input'])


    def _log_loss(self, args, vals):
        """Add the Jacobian adjustment if input is a distribution."""
        if isinstance(self.arg['input'], BaseModel):
            # TODO: compute the jacobian adjustment
            pass
        else:
            return 0

    # TODO: hmm, well if you can compute the log loss of it then... maybe these 
    # transformations CAN inherit from model instead of layer


class Log(BaseLayer):
    """A layer which outputs the natural log of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.log(args['input'])


    def _log_loss(self, obj, vals):
        """The loss is a Jacobian adjustment if input is a distribution."""
        if isinstance(self.arg['input'], BaseModel):
            # TODO: compute the jacobian adjustment
            pass
        else:
            return 0



# TODO: Logit



# TODO: Probit



# TODO: Softmax (?)



class Dense(BaseLayer):
    """A densely-connected neural network layer.


    TODO: More info...


    """


    # Layer arguments and their default values
    _default_args = {
        'input': None, 
    }


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'units': 1, 
        'activation': None, 
        'use_bias': True,
        'weight_initializer': None, #TODO: glorot or something as default?
        'bias_initializer': None,   #TODO: glorot or something as default?
        'weight_prior': Normal(0, 1),
        'bias_prior': Normal(0,1),
    }
    

    def _build(self, args, data):
        """Build the layer."""


        # If no input specified, assume data is input
        if isinstance(args['input'], np.ndarray) and len(args['input'])==0:
            self.input = data

        # TODO

        # NOTE: may have to implement manually w/ probflow.Variable? in order to let the mean_model work...


    def _log_loss(self, obj, vals):
        """Log loss incurred by this layer."""
        # TODO
        pass


    def _kl_loss(self, obj, vals):
        """The sum of divergences of variational posteriors from priors."""
        # TODO
        pass



class Sequential(BaseLayer):
    """A sequence of layers.


    TODO: More info...


    """

    # TODO
    pass



class Conv1d(BaseLayer):
    """A 1-dimensional convolutional neural network layer.


    TODO: More info...


    """

    # TODO
    pass



class Conv2d(BaseLayer):
    """A 2-dimensional convolutional neural network layer.


    TODO: More info...


    """

    # TODO
    pass



# TODO: Pooling layer



class Embedding(BaseLayer):
    """A categorical embedding layer.


    TODO: More info...


    """

    # TODO
    pass


# TODO: LSTM