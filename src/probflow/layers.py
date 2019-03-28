"""Layers.

TODO: more info...
A layer, unlike a model, returns a single value
whereas a model returns a probability distribution!


Data Layers
-----------

* :class:`.Input`


Basic Arithmetic Layers
-----------------------

TODO: mention that these are all elementwise

* :class:`.Add`
* :class:`.Sub`
* :class:`.Mul`
* :class:`.Div`
* :class:`.Neg`
* :class:`.Abs`
* :class:`.Exp`
* :class:`.Log`
* :class:`.Reciprocal`
* :class:`.Sqrt`


Transformation Layers
---------------------

TODO: mention that these don't change the shape

* :class:`.Sigmoid`
* :class:`.Relu`
* :class:`.Softmax`


Reduce Layers
-------------

TODO: mention that these DO change the shape

* :class:`.Sum`
* :class:`.Mean`
* :class:`.Min`
* :class:`.Max`
* :class:`.Prod`
* :class:`.LogSumExp`


Matrix Layers
-------------

* :class:`.Cat`
* :class:`.Dot`
* :class:`.Matmul`


Neural Network Layers
---------------------

* :class:`.Dense`
* :class:`.BatchNormalization`
* :class:`.Sequential`
* :class:`.Gather`
* :class:`.Embedding`


Custom Transform Layer
----------------------

A layer which performs an elementwise transform using any arbitrairy |TensorFlow| ops. 

* :class:`.Transform`

----------

"""

__all__ = [
    'Input',
    'Add',
    'Sub',
    'Mul',
    'Div',
    'Neg',
    'Abs',
    'Exp',
    'Log',
    'Reciprocal',
    'Sqrt',
    'Transform',
    'Sigmoid',
    'Relu',
    'Softmax',
    'Sum',
    'Mean',
    'Min',
    'Max',
    'Prod',
    'LogSumExp',
    'Cat',
    'Dot',
    'Matmul',
    'Dense',
    'BatchNormalization',
    'Sequential',
    'Gather',
    'Embedding',
]

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from .core import BaseLayer, REQUIRED
from .distributions import Normal
from .parameters import Parameter



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


    # Integer column ids
    _int_cols = None


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if (kwargs['cols'] is not None and 
                not isinstance(kwargs['cols'], (int, str, list))):
            raise ValueError('cols kwarg must be None, int, str, ' +
                             'or list of int or str')


    def _build(self, _args, data, _batch_shape):
        """Build the layer."""
        if self.kwargs['cols'] is None:
            return data
        else:
            if self._int_cols is None:
                raise RuntimeError('Integer columns were not set for Input')
            return tf.transpose(tf.gather(tf.transpose(data), self._int_cols))


    def __str__(self, prepend=''):
        """String representation of a parameter."""
        if self.kwargs['cols'] is None:
            return 'Input (all columns)'
        else:
            return 'Input '+str(self.kwargs['cols'])



class Add(BaseLayer):
    """A layer which adds two inputs.


    TODO: More info... (elementwise)


    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return args['a'] + args['b']



class Sub(BaseLayer):
    """A layer which subtracts one input from another.


    TODO: More info... (elementwise)


    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return args['a'] - args['b']



class Mul(BaseLayer):
    """A layer which multiplies two inputs.


    TODO: More info... (elementwise)


    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        # TODO: have to distinguish between matrix and elementwise mult somehow?
        return args['a'] * args['b']



class Div(BaseLayer):
    """A layer which divides one input by another.


    TODO: More info... (elementwise)


    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return args['a'] / args['b']



class Neg(BaseLayer):
    """A layer which outputs the negative of its input.


    TODO: More info... (elementwise)


    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return -args['input']



class Abs(BaseLayer):
    """A layer which outputs the absolute value of its input.


    TODO: More info... (elementwise)


    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return abs(args['input'])



class Exp(BaseLayer):
    r"""A layer which outputs the natural exponent of its input.

    TODO: More info...

    Given :math:`x`, this layer returns :math:`\exp x`, elementwise.

    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.exp(args['input'])



class Log(BaseLayer):
    r"""A layer which outputs the natural log of its input.


    TODO: More info...

    Given :math:`x`, this layer returns :math:`\log x`, elementwise.

    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.log(args['input'])



class Reciprocal(BaseLayer):
    r"""A layer which outputs the reciprocal of its input.


    TODO: More info...

    Given :math:`x`, this layer returns (elementwise):

    .. math::

        \text{Reciprocal}(x) = \frac{1}{x}

    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reciprocal(args['input'])



class Sqrt(BaseLayer):
    r"""A layer which outputs the square root of its input.


    TODO: More info...

    Given :math:`x`, this layer returns (elementwise):

    .. math::

        \text{Sqrt}(x) = \sqrt{x}

    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.sqrt(args['input'])



class Transform(BaseLayer):
    r"""Performs an elementwise transform using arbitrairy |TensorFlow| ops. 


    TODO: More info...

    Given :math:`x`, and some function :math:`f`, this layer returns 
    :math:`f(x)`, elementwise.

    """

    # Layer keyword arguments and their default values
    _default_kwargs = {
        'func': lambda x: x,
    }

    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not callable(kwargs['func']):
            raise ValueError('func kwarg must be a callable')

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return self.kwargs['func'](args['input'])



class Sigmoid(BaseLayer):
    r"""A layer which passes its input through a sigmoid function, elementwise.


    TODO: More info...

    Given :math:`x`, this layer returns (elementwise):

    .. math::

        \text{Sigmoid}(x) = \frac{1}{1 + \exp (-x)}

    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.sigmoid(args['input'])



class Relu(BaseLayer):
    r"""A layer which linearly rectifies its input.


    TODO: More info...

    Given :math:`x`, this layer returns (elementwise):

    .. math::

        \text{Relu}(x) = \max (x, 0)

    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.nn.relu(args['input'])



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


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.nn.softmax(args['input'], axis=self.kwargs['axis'])



class Sum(BaseLayer):
    r"""A layer which outputs the sum of its inputs.


    TODO: More info...

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{Sum}(\mathbf{x}) = \sum_i x_i

    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_sum(args['input'], 
                             axis=self.kwargs['axis'], 
                             keepdims=True)



class Mean(BaseLayer):
    r"""A layer which outputs the mean of its inputs.


    TODO: More info...

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{Mean}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N x_i

    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_mean(args['input'], 
                              axis=self.kwargs['axis'],
                              keepdims=True)



class Min(BaseLayer):
    """A layer which outputs the minimum of its inputs.


    TODO: More info...


    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_min(args['input'], 
                             axis=self.kwargs['axis'],
                             keepdims=True)



class Max(BaseLayer):
    """A layer which outputs the maximum of its inputs.


    TODO: More info...


    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_max(args['input'], 
                             axis=self.kwargs['axis'],
                             keepdims=True)



class Prod(BaseLayer):
    r"""A layer which outputs the product of its inputs.


    TODO: More info...

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{Sum}(\mathbf{x}) = \prod_i x_i

    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_prod(args['input'], 
                              axis=self.kwargs['axis'],
                              keepdims=True)



class LogSumExp(BaseLayer):
    r"""A layer which outputs the log(sum(exp(inputs))).


    TODO: More info...

    TODO: explain why this is useful when working in log space, numerical stability etc

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{LogSumExp}(\mathbf{x}) = \log \left( \sum_i \exp x_i \right)

    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_logsumexp(args['input'], 
                                   axis=self.kwargs['axis'],
                                   keepdims=True)



class Cat(BaseLayer):
    """A layer which Concatenates its two inputs.


    TODO: More info...

    TODO: really we want to be able to pass a LIST of inputs, not just 2

    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.concat([args['a'], args['b']], axis=self.kwargs['axis'])



class Dot(BaseLayer):
    r"""A layer which outputs the dot product of its two inputs.


    TODO: More info...

    Given a two vectors :math:`\mathbf{a}` and :math:`\mathbf{b}`,
    this layer returns:

    .. math::

        \text{Dot}(\mathbf{a},\mathbf{b}) =
        \mathbf{a} \cdot \mathbf{b} =
        \sum_i ( a_i b_i )

    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        # TODO: this will only work w/ vector inputs...
        return tf.reduce_sum(args['a'] * args['b'], 
                             axis=self.kwargs['axis'],
                             keepdims=True)



class Matmul(BaseLayer):
    """A layer which outputs the matrix multiplication of its two inputs.


    TODO: More info...


    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.matmul(args['a'], args['b'])
        # TODO: don't think this will work w/ tensors of >2 dims...



# TODO: Inverse - inverts a matrix



class Dense(BaseLayer):
    """A densely-connected neural network layer.


    TODO: More info...


    """


    # Layer arguments and their default values
    _default_args = {
        'input': Input(),
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


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        # TODO
        pass


    def _build(self, args, data, batch_shape):
        """Build the layer."""

        # Inputs
        x_in = args['input']
        ndims = x_in.shape[1]
        units = self.kwargs['units']

        # TODO: make a scope for this layer's variables

        # Create weight and bias parameters
        weight = Parameter(shape=[ndims, units],
                           prior=self.kwargs['weight_prior'])
        bias = Parameter(shape=[1, units],
                         prior=self.kwargs['bias_prior'])

        # Build the weight and bias parameter
        weight.build(data, batch_shape)
        bias.build(data, batch_shape)

        # Compute output using a sample from the variational posteriors
        weight_samples = self.weight.built_obj
        bias_samples = self.bias.built_obj
        y_out = tf.matmul(x_in, weight_samples) + bias_samples
        self._sample = self.kwargs['activation'](y_out)

        # Compute the output using the means of the variational posteriors
        weight_means = self.weight.mean_obj
        bias_means = self.bias.mean_obj
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


    def _log_loss(self, vals):
        """Log loss incurred by this layer."""
        return self._log_loss_sum


    def _mean_log_loss(self, vals):
        """Log loss incurred by this layer w/ mean parameters."""
        return self._mean_log_loss_sum


    def _kl_loss(self):
        """The sum of divergences of variational posteriors from priors."""
        return self._kl_loss_sum



class BatchNormalization(BaseLayer):
    r"""A layer which normalizes its inputs.

    Batch normalization is a technique which normalizes and re-scales and 
    offsets the output of one layer before passing it on to another layer
    [1]_.  It often leads to faster training of neural networks, and better
    generalization error by stabilizing the change in the layers' input
    distributions, or perhaps by smoothing the optimization landscape [2]_.

    Given a set of tensors for this batch, where :math:`x_{ij}` is
    the :math:`i`-th element of the :math:`j`-th sample in this batch,
    this layer returns an elementwise transformation of the input tensors
    according to:

    .. math::

        \text{BatchNorm}(x_{ij}) = 
        \gamma_i \left( \frac{x_{ij} - \mu_i}{\sigma_i} \right)
        + \beta_i

    Where :math:`\mu_i` is the mean of the :math:`i`-th element:

    .. math::

        \mu_i = \frac{1}{N} \sum_{k=1}{N} \mathbf x_{ik}

    and :math:`\sigma_i` is the standard deviation of the :math:`i`-th 
    element:

    .. math::

        \mu_i = \frac{1}{N} \sum_{k=1}{N} (\mathbf x_{ik} - \mu_i)^2

    and :math:`\gamma` and :math:`\beta` are two free parameters for each 
    element.

    Parameters
    ----------
    input : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Input to batch-normalize.

    Keyword Arguments
    -----------------
    weight_posterior : |Distribution|
        Probability distribution class to use to approximate the posterior
        for the weight parameter(s) (:math:`\gamma`).
        Default = :class:`.Normal`
    bias_posterior : |Distribution|
        Probability distribution class to use to approximate the posterior
        for the bias parameter(s) (:math:`\beta`).
        Default = :class:`.Normal`
    weight_prior : |None| or a |Distribution| object
        Prior probability distribution for the weight parameter(s)
        (:math:`\gamma`).  |None| or a |Distribution| function which has been
        instantiated with parameters.
        Default = :class:`.Normal` ``(0,1)``
    bias_prior : |None| or a |Distribution| object
        Prior probability distribution for the bias parameter(s)
        (:math:`\beta`).  |None| or a |Distribution| function which has been
        instantiated with parameters.
        Default = :class:`.Normal` ``(0,1)``
    weight_initializer : {|None| or dict or |Tensor| or |Initializer|}
        Initializer for each of the weights' variational posterior parameters.
        See :class:`.Parameter` for more information on how to specify a
        custom ``initializer``.
    bias_initializer : {|None| or dict or |Tensor| or |Initializer|}
        Initializer for each of the weights' variational posterior parameters.
        See :class:`.Parameter` for more information on how to specify a
        custom ``initializer``.

    Examples
    --------

    .. code-block:: python

        x = Input()
        l1 = Dense(x, units=128)
        l1_norm = BatchNormalization(layer1)
        l2 = Dense(l1, units=64)
        ...

    References
    ----------
    .. [1] Sergey Ioffe and Christian Szegedy.
        Batch Normalization: Accelerating Deep Network Training by
        Reducing Internal Covariate Shift.
        *arXiv preprint*, 2015. http://arxiv.org/abs/1502.03167
    .. [2] Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and 
        Aleksander Madry. How Does Batch Normalization Help Optimization?
        *arXiv preprint*, 2018. http://arxiv.org/abs/1805.11604
    """

    # Layer arguments and their default values
    _default_args = {
        'input': Input(),
    }


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'weight_posterior': Normal,
        'bias_posterior': Normal,
        'weight_initializer': None,
        'bias_initializer': None,
        'weight_prior': Normal(0, 1),
        'bias_prior': Normal(0, 1),
    }


    def _build(self, args, data, batch_shape):
        """Build the layer."""

        # Inputs
        x_in = args['input']
        dims = x_in.shape[1:]

        # Normalize input elementwise
        x_mean = tf.reduce_mean(x_in, axis=0, keepdims=True)
        x_std = tf.nn.moments(x_in, axes=[0], keepdims=True)
        x_norm = (x_in-x_mean)/x_std

        # TODO: make a scope for this layer's variables

        # Create weight and bias parameters
        weight = Parameter(shape=dims,
                           posterior_fn=self.kwargs['weight_prior'],
                           initializer=self.kwargs['weight_initializer'],
                           prior=self.kwargs['weight_prior'])
        bias = Parameter(shape=dims,
                         posterior_fn=self.kwargs['bias_prior'],
                         initializer=self.kwargs['bias_initializer'],
                         prior=self.kwargs['bias_prior'])

        # Build the weight and bias parameter
        weight.build(data, batch_shape)
        bias.build(data, batch_shape)

        # Compute output using a sample from the variational posteriors
        weight_samples = self.weight.built_obj
        bias_samples = self.bias.built_obj
        self._sample = x_norm*weight_samples + bias_samples

        # Compute the output using the means of the variational posteriors
        weight_means = self.weight.mean_obj
        bias_means = self.bias.mean_obj
        self._mean = x_norm*weight_means + bias_means

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


    def _log_loss(self, vals):
        """Log loss incurred by this layer."""
        return self._log_loss_sum


    def _mean_log_loss(self, vals):
        """Log loss incurred by this layer w/ mean parameters."""
        return self._mean_log_loss_sum


    def _kl_loss(self):
        """The sum of divergences of variational posteriors from priors."""
        return self._kl_loss_sum



class Sequential(BaseLayer):
    """A sequence of layers.


    TODO: More info...


    """


    # Layer arguments and their default values
    _default_args = {
        'input': Input(),
    }


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'layers': [],
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['layers'], list):
            raise ValueError('layers kwarg must be a list of layers')
        for layer in kwargs['layers']:
            if not isinstance(kwargs['layers'], BaseLayer):
                raise ValueError('each element of layers must be a BaseLayer')
            if len(layer._default_args) > 1:
                raise RuntimeError('each layer must take only 1 input')


    def _build(self, args, data, batch_shape):
        """Build the layer."""

        # List of layers
        layers = kwargs['layers']

        # Connect the layers
        output = args['input']
        for layer in layers:
            layer.args['input'] = output
            output = layer

        # Store a list of all parameters in each layer
        for layer in layers:
            self._parameters = layer._parameter_list()

        # Build the layers
        layers[-1].build(data, batch_shape)

        # Store mean and sample
        self._sample = layers[-1]._sample
        self._mean = layers[-1]._mean

        # Store the losses
        self._log_loss_sum = layers[-1].samp_loss_sum
        self._mean_log_loss_sum = layers[-1].mean_loss_sum
        self._kl_loss_sum = layers[-1].kl_loss_sum

        # Return the sample from the last layer
        return self._sample


    def _build_mean(self, args, data):
        """Build the layer with mean parameters.

        TODO: docs
        Note that this was done in _build()

        """
        return self._mean


    def _log_loss(self, vals):
        """Log loss incurred by this layer."""
        return self._log_loss_sum


    def _mean_log_loss(self, vals):
        """Log loss incurred by this layer w/ mean parameters."""
        return self._mean_log_loss_sum


    def _kl_loss(self):
        """The sum of divergences of variational posteriors from priors."""
        return self._kl_loss_sum



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



class Gather(BaseLayer):
    """Collects slices from a layer using indexes provided by another layer.

    TODO: docs

    e.g. if 
    vals = [1.1, 2.2, 3.3, 4.4]  (a 4x1 vector), and
    inds = [0, 2, 1]
    then Gather(vals, inds) returns:
    [1.1, 3.3, 2.2] (a 3x1 vector)

    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('values', REQUIRED),
        ('indices', REQUIRED)
    ])


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': 0,
        'index_dtype': tf.uint32,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')
        if not isinstance(kwargs['index_dtype'], tf.DType):
            raise ValueError('index_dtype kwarg must be a tf.DType')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.gather(args['values'], 
                         tf.cast(args['indexes'], self.kwargs['index_dtype']), 
                         axis=self.kwargs['axis'])



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

    should cast input to int64 or something before using as indexes -> tf.gather

    same idea as Gather (above), just it creates the embedding array for you
    based on how many unique vals are in the training dataset

    will somehow need to be made aware of how many unique values there are in
    the column(s) the layer is embedding...
    takes a kwarg="unique_vals" so it knows how many embedding params to create?
    no that would be redundant for the user...
    might just have to, from fit, recursively go thru the model and give each
    embedding layer access to the unique values of their cols

    also should take a dims kwarg (embedding dimensions)

    """

    # TODO
    pass


# TODO: LSTM
