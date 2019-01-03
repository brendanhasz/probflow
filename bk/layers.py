"""Layers.

TODO: more info...

A layer, unlike a model, returns a single value
whereas a model returns a probability distribution!

"""

import numpy as np

from .distributions import Normal
from .core import BaseLayer



class Input(BaseLayer):
    """Layer which represents the input data.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'cols': np.array([]) #indexes of columns to use
    }
    

    def _build(self, args, data):
        """Build the layer."""
        if isinstance(args['cols'], np.ndarray) and len(args['cols'])==0:
            return data
        else:
            # TODO: slice data by cols
            pass


    def _log_loss(self, obj, vals):
        """Data incurrs no loss."""
        return 0



class Add(BaseLayer):
    """A layer which adds two inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'a': None,
        'b': None
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return args['a'] + args['b']


    def _log_loss(self, obj, vals):
        """Addition incurrs no loss."""
        return 0



class Sub(BaseLayer):
    """A layer which subtracts one input from another.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'a': None,
        'b': None
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return args['a'] - args['b']


    def _log_loss(self, obj, vals):
        """Subtraction incurrs no loss."""
        return 0



class Mul(BaseLayer):
    """A layer which multiplies two inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'a': None,
        'b': None
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return args['a'] * args['b']


    def _log_loss(self, obj, vals):
        """Multiplication incurrs no loss."""
        return 0



class Div(BaseLayer):
    """A layer which divides one input by another.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'a': None,
        'b': None
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return args['a'] / args['b']


    def _log_loss(self, obj, vals):
        """Division incurrs no loss."""
        return 0



class Abs(BaseLayer):
    """A layer which outputs the absolute value of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'val': None
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.abs(args['val'])


    def _log_loss(self, obj, vals):
        """Absolute value incurrs no loss."""
        return 0



class Exp(BaseLayer):
    """A layer which outputs the natural exponent of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'val': None
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.exp(args['val'])


    def _log_loss(self, args, vals):
        """Add the Jacobian adjustment if input is a distribution."""
        if isinstance(self.arg['val'], BaseModel):
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
        'val': None
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.log(args['val'])


    def _log_loss(self, obj, vals):
        """The loss is a Jacobian adjustment if input is a distribution."""
        if isinstance(self.arg['val'], BaseModel):
            # TODO: compute the jacobian adjustment
            pass
        else:
            return 0



# TODO: Logit



# TODO: Probit



class Dense(BaseLayer):
    """A densely-connected neural network layer.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': np.array([]), 
        'units': 1, 
        'activation': np.array([]), 
        'use_bias': True,
        'weight_initializer': None, #TODO: glorot or something as default?
        'bias_initializer': None,   #TODO: glorot or something as default?
        'weight_prior_fn': Normal,
        'weight_prior_args': [0, 1],
        'bias_prior_fn': Normal,
        'bias_prior_args': [0, 1]
    }
    

    def _build(self, args, data):
        """Build the layer."""


        # If no input specified, assume data is input
        if isinstance(args['input'], np.ndarray) and len(args['input'])==0:
            self.input = data


        # TODO

        # NOTE: may have to implement manually w/ bk.Variable? in order to let the mean_model work...


# TODO: _default_args should really only be for *tensor* (or tensor-generating)
# args, may have to have a separate _default_kwargs or something for, eg, 
# units, activation, etc... (everything except input, really...)
# cols arg to Input - same idea, that's not a tensor, thats a list of ints


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