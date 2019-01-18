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
        if isinstance(args['cols'], np.ndarray) and len(args['cols']) == 0:
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



class Sigmoid(BaseLayer):
    r"""A layer which passes its input through a sigmoid function, elementwise.


    TODO: More info...

    Given :math:`x`, this layer returns:

    .. math::

        \text{Sigmoid}(x) = \frac{1}{1 + \exp (-x)}

    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.sigmoid(args['input'])



class Softmax(BaseLayer):
    r"""A layer which outputs the softmax of its input.


    TODO: More info...

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{Softmax}(\mathbf{x}) = \mathbf{\sigma}

    where

    .. math::

        \sigma_i = \frac{\exp (x_i)}{\sum_j \exp (x_j)}

    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }

    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': 1,
    }    

    def _build(self, args, data):
        """Build the layer."""
        return tf.nn.softmax(args['input'], axis=self.kwargs['axis'])



# TODO: Logit



# TODO: Probit



# TODO: Softmax (?)



class Sum(BaseLayer):
    """A layer which outputs the sum of its inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.reduce_sum(args['input'])



class Mean(BaseLayer):
    """A layer which outputs the mean of its inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.reduce_mean(args['input'])



class Max(BaseLayer):
    """A layer which outputs the maximum of its inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.reduce_max(args['input'])



class Min(BaseLayer):
    """A layer which outputs the minimum of its inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.reduce_min(args['input'])



class Prod(BaseLayer):
    """A layer which outputs the product of its inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.reduce_prod(args['input'])



class LogSumExp(BaseLayer):
    """A layer which outputs the log(sum(exp(inputs))).


    TODO: More info...

    TODO: explain why this is useful when working in log space, numerical stability etc


    """

    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.reduce_logsumexp(args['input'])



class Dot(BaseLayer):
    """A layer which outputs the dot product of its two inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])
    

    def _build(self, args, data):
        """Build the layer."""
        return tf.reduce_sum(args['a'] * args['b'], axis=1)
        # TODO: this will only work w/ vector inputs...



# TODO: MatMul? for stuff of size batch_size-by-dim1-by-dim2



class Cat(BaseLayer):
    """A layer which Concatenates its two inputs.


    TODO: More info...


    """

    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])

    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': 1,
    }

    def _build(self, args, data):
        """Build the layer."""
        return tf.concat(args['a'], args['b'], axis=self.kwargs['axis'])



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
        'activation': tf.nn.relu, 
        'weight_initializer': None, #TODO: glorot or something as default?
        'bias_initializer': None,   #TODO: glorot or something as default?
        'weight_prior': Normal(0, 1),
        'bias_prior': Normal(0, 1),
    }
    

    def _build(self, args, data):
        """Build the layer."""

        # If no input specified, assume data is input
        if args['input'] is None:
            x_in = data
        else:
            x_in = args['input']

        # Compute dimensions
        # batch_size = x_in.shape[0]
        ndims = x_in.shape[1]
        units = self.kwargs['units']

        # Create weight and bias variables
        weight = Variable(shape=[ndims, units], 
                          prior=self.kwargs['weight_prior'])
        bias = Variable(shape=[1, units],
                        prior=self.kwargs['bias_prior'])

        # Build the weight and bias variables
        weight._build(data)
        bias._build(data)

        # Compute output using a sample from the variational posteriors
        weight_samples = self.weight._sample(data)
        bias_samples = self.bias._sample(data)
        y_out = tf.matmul(x_in, weight_samples) + bias_samples
        self._sample = self.kwargs['activation'](y_out)

        # Compute the output using the means of the variational posteriors
        weight_means = self.weight._mean(data)
        bias_means = self.bias._mean(data)
        mean_y_out = tf.matmul(x_in, weight_means) + bias_means
        self._mean = self.kwargs['activation'](mean_y_out)

        # Compute the losses
        self._log_loss_sum = (weight._log_loss(weight_samples) +
                              bias._log_loss(bias_samples))
        self._mean_log_loss_sum = (weight._log_loss(weight_means) +
                                   bias._log_loss(bias_means))
        self._kl_loss_sum = weight._kl_loss() + bias._kl_loss()

        # Return the sample
        return self._sample


    def _build_mean(self, args, data):
        """Build the layer with mean parameters.

        TODO: docs
        Note that this was done in _build()

        """
        return self._mean


    def _log_loss(self, obj, vals):
        """Log loss incurred by this layer."""
        return self._log_loss_sum


    def _mean_log_loss(self, obj, vals):
        """Log loss incurred by this layer w/ mean parameters."""
        return self._mean_log_loss_sum


    def _kl_loss(self, obj, vals):
        """The sum of divergences of variational posteriors from priors."""
        return self._kl_loss_sum



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

    TODO: prior/regularization for the embedding layer might be difficult b/c
    we're not including the log posterior in the loss (just the divergence and
    the likelihood).  We just want point estimates for the embeddings (not 
    variational posterior dist estimates) b/c of the rotational/multimodal 
    posterior problem.  But w/o a distribution, can't compute the divergence
    between a point and a distribution.  May have to pretend that there are
    variance-1 normal dists around the embedding points and compute the KL div 
    using that.


    """

    # TODO
    pass


# TODO: LSTM