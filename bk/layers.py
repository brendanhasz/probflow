"""Layers.

TODO: more info...

"""



from abc import ABC, abstractmethod



class BaseLayer(ABC):
    """Abstract layer class (just used as an implementation base)


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



class Input(BaseLayer):
    """Layer which represents the input data.


    TODO: More info...


    """

    # TODO



class Sequential(BaseLayer):
    """A sequence of layers.


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