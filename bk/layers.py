"""Layers.

TODO: more info...

"""



from abc import ABC, abstractmethod



class BaseLayer(ABC):
    """Abstract layer class (just used as an implementation base)


    TODO: More info...


    """

    # TODO: i feel like BaseLayer should have everything BaseModel has,
    # up until the fit() method.  So maybe should put BaseLayer in 
    # base_models.py, and then BaseModel inherits BaseLayer and adds to it 
    # (adds fit, predict, all that stuff) 

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

        Also has to update self.log_loss
        """
        pass        



class Input(BaseLayer):
    """Layer which represents the input data.


    TODO: More info...


    """

    # TODO


class Add(BaseLayer):
    """A layer which adds two inputs.


    TODO: More info...


    """

    def __init__(self, a, b):
        """Construct the layer.

        TODO: docs...

        """
        pass


    def build(self, data):
        """Build the tensorflow graph for this layer and its arguments.

        TODO: docs...

        TODO: Also has to update self.log_loss
        """
        pass    

    if isinstance(input, (int, float, np.ndarray)):
        return np.exp(input)
    elif isinstance(input, tf.Tensor):
        return tf.exp(input)
    elif isinstance(input, BaseModel):
        return ExpTransform(input)
    elif isinstance(input, BaseLayer):

    elif isinstance(input, BaseTransformation):

    else:


class Sub(BaseLayer):
    """A layer which subtracts one input from another.


    TODO: More info...


    """

    # TODO


class Mul(BaseLayer):
    """A layer which multiplies two inputs.


    TODO: More info...


    """

    # TODO


class Div(BaseLayer):
    """A layer which divides one input by another.


    TODO: More info...


    """

    # TODO


class Exp(BaseLayer):
    """A layer which outputs the natural exponent of its input.


    TODO: More info...


    """

    # TODO


class Log(BaseLayer):
    """A layer which outputs the natural log of its input.


    TODO: More info...


    """

    # TODO


class Dense(BaseLayer):
    """A densely-connected neural network layer.


    TODO: More info...


    """

    def __init__(self, 
                 input=None, 
                 units=1, 
                 activation=None, 
                 use_bias=True,
                 weight_initializer=??,
                 bias_initializer=??,
                 weight_prior=??Normal(0,1)?,
                 bias_prior=??Normal(0,1)?):
        """Construct the dense layer.

        TODO: docs...

        """

        # Check inputs are of correct type
        # TODO

        # Assign attributes
        self.input = input
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior


    def build(self, data):
        """Build the tensorflow graph for this layer and its arguments.

        TODO: docs...

        """

        # If no input specified, assume data is input
        if self.input is None:
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