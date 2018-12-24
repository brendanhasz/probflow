"""Layers.

TODO: more info...

A layer, unlike a model, returns a single value
whereas a model returns a probability distribution!

"""



from core import BaseLayer



class Input(BaseLayer):
    """Layer which represents the input data.


    TODO: More info...


    """

    # TODO


class Add(BaseLayer):
    """A layer which adds two inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    self.default_args = {
        'a': None,
        'b': None
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """
        return args['a'] + args['b']


class Sub(BaseLayer):
    """A layer which subtracts one input from another.


    TODO: More info...


    """

    # Layer arguments and their default values
    self.default_args = {
        'a': None,
        'b': None
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """
        return args['a'] - args['b']



class Mul(BaseLayer):
    """A layer which multiplies two inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    self.default_args = {
        'a': None,
        'b': None
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """
        return args['a'] * args['b']



class Div(BaseLayer):
    """A layer which divides one input by another.


    TODO: More info...


    """

    # Layer arguments and their default values
    self.default_args = {
        'a': None,
        'b': None
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """
        return args['a'] / args['b']



class Abs(BaseLayer):
    """A layer which outputs the absolute value of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    self.default_args = {
        'val': None
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """
        return tf.abs(args['val'])



class Exp(BaseLayer):
    """A layer which outputs the natural exponent of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    self.default_args = {
        'val': None
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """
        return tf.exp(args['val'])



class Log(BaseLayer):
    """A layer which outputs the natural log of its input.


    TODO: More info...


    """

    # Layer arguments and their default values
    self.default_args = {
        'val': None
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """
        return tf.log(args['val'])



class Dense(BaseLayer):
    """A densely-connected neural network layer.


    TODO: More info...


    """

    # Layer arguments and their default values
    self.default_args = {
        'input': [], 
        'units': 1, 
        'activation': [], 
        'use_bias': True,
        'weight_initializer': ??,
        'bias_initializer': ??,
        'weight_prior': ??Normal(0,1)??,
        'bias_prior': ??Normal(0,1)??
    }
    

    def _build(self, args, data):
        """Build the distribution model.

        TODO: Docs...

        """

        # If no input specified, assume data is input
        if isinstance(args['input'], list) and len(args['input'])==0:
            self.input = data


        # TODO
        # NOTE: may have to implement manually w/ bk.Variable? in order to let the mean_model work...

        # TODO: should return built_model, mean_model



class Sequential(BaseLayer):
    """A sequence of layers.


    TODO: More info...


    """

    # TODO



class Conv1d(BaseLayer):
    """A 1-dimensional convolutional neural network layer.


    TODO: More info...


    """

    # TODO



class Conv2d(BaseLayer):
    """A 2-dimensional convolutional neural network layer.


    TODO: More info...


    """

    # TODO



# TODO: Pooling layer



class Embedding(BaseLayer):
    """A categorical embedding layer.


    TODO: More info...


    """

    # TODO



# TODO: LSTM