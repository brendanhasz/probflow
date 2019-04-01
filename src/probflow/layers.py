"""Layers.

Layers take one or more tensors as input arguments, perform some computation
on them, and output a single tensor.  They can additionally take keyword
arguments which control how the computation is performed.


Data Layers
-----------

The data input layer represents input(s) to the model, and allows for
selecting a subset of the input data to pass through a specific part of the
model.

* :class:`.Input`


Basic Arithmetic Layers
-----------------------

These layers perform element-wise arithmetic given two input tensors.
The two input tensors must be the same size, and the output is the same 
size as one of the inputs.

* :class:`.Add`
* :class:`.Sub`
* :class:`.Mul`
* :class:`.Div`


Transformation Layers
---------------------

These layers perform element-wise transformations on a single input tensor.
The output is the same size as the input.

* :class:`.Neg`
* :class:`.Abs`
* :class:`.Exp`
* :class:`.Log`
* :class:`.Reciprocal`
* :class:`.Sqrt`
* :class:`.Sigmoid`
* :class:`.Relu` - rectified linear unit activation function
* :class:`.Softmax`
* :class:`.Transform` - define a custom transformation


Reduce Layers
-------------

These layers perform reduction operations on a single input tensor.
Unlike the above Transformation layers, reduce layers *do* change the shape
of the output relative to the input.  For example, if given a NxM tensor,
:class:`.Sum` will return a Nx1 tensor, having taken the sum across each row.

* :class:`.Sum`
* :class:`.Mean`
* :class:`.Min`
* :class:`.Max`
* :class:`.Prod`
* :class:`.LogSumExp`


Matrix Layers
-------------

These layers perform matrix- and vector-related operations.

* :class:`.Cat` - concatenate vectors/matrixes
* :class:`.Dot` - dot product
* :class:`.Matmul` - matrix multiplication


Neural Network Layers
---------------------

These layers perform various neural-network-related operations.  Some of them,
including :class:`.Dense`, :class:`.BatchNormalization`, and 
:class:`.Embedding` add new :class:`.Parameter` s to the model.

* :class:`.Dense` - fully-connected neural network layer
* :class:`.BatchNormalization` - normalize data per batch
* :class:`.Sequential` - apply a list of layers sequentially
* :class:`.Gather` - look up values based on some index
* :class:`.Embedding` - embed categorical data in a lower-dimensional space

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

from .core import BaseLayer, BaseDistribution, REQUIRED
from .distributions import Normal, Deterministic
from .parameters import Parameter



def _validate_initializer(initializer):
    """Ensure an object is a valid initializer"""
    init_types = (dict, np.ndarray, tf.Tensor, 
                  tf.keras.initializers.Initializer)
    if initializer is not None and not isinstance(initializer, init_types):
        raise TypeError('initializer must be None, a Tensor, an'
                        ' Initializer, or a dict')
    if isinstance(initializer, dict):
        for arg in initializer:
            if (initializer[arg] is not None and
                not isinstance(initializer[arg], init_types)):
                raise TypeError('each value in initializer dict must be '
                                'None, a Tensor, or an Initializer')



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
            return tf.gather(data, self._int_cols, axis=1)


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
    r"""A fully-connected neural network layer.

    Given a vector :math:`\mathbf{x}`, this layer first performs a linear
    transformation on :math:`\mathbf{x}` by matrix-multiplying 
    :math:`\mathbf{x}` by a weight matrix :math:`\mathbf{W}` and adding 
    a bias :math:`\mathbf{b}`, and then passes the result of that linear 
    transformation through some non-linear, elementwise activation function 
    :math:`f`:

    .. math::

        \text{Dense}(\mathbf{x}) = f(\mathbf{x}^\top \mathbf{W} + \mathbf{b})

    Where :math:`\mathbf{W}` is an :math:`N \times M` matrix of 
    parameters, :math:`\mathbf{b}` is a :math:`M`-length vector of parameters,
    :math:`N` is the length of :math:`\mathbf{x}`, and :math:`M` is the length
    of the output vector.

    Any function can be specified for :math:`f` using the ``activation`` 
    keyword argument, but the default is to use a the rectified linear unit
    activation function:

    .. math::

        f(x) = \max (0, x)

    The number of input dimensions (:math:`N`) is automatically determined by
    the shape of the ``input``, and the number of output dimensions can be set
    using the ``units`` keyword argument.

    TODO: diagram


    Parameters
    ----------
    input : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Input to pass through the fully-connected neural network layer.
        The default is to use all the input dimensions.


    Keyword Arguments
    -----------------
    units : int
        Number of units in the output layer (:math:`M`).
        Default = 1.
    name : str
        Name for this layer.
        Default = 'Dense'
    activation : callable
        Activation function to apply after the linear transformation.
        Default = ``tf.nn.relu`` (rectified linear unit)
    weight_posterior : |Distribution|
        Probability distribution class to use to approximate the posterior
        for the weight parameter(s) (:math:`\mathbf{W}`).
        Default = :class:`.Normal`
    bias_posterior : |Distribution|
        Probability distribution class to use to approximate the posterior
        for the bias parameter(s) (:math:`\mathbf{b}`).
        Default = :class:`.Normal`
    weight_prior : |None| or a |Distribution| object
        Prior probability distribution for the weight parameter(s)
        (:math:`\mathbf{W}`).  |None| or a |Distribution| function which has 
        been instantiated with parameters.
        Default = :class:`.Normal` ``(0,1)``
    bias_prior : |None| or a |Distribution| object
        Prior probability distribution for the bias parameter(s)
        (:math:`\mathbf{b}`).  |None| or a |Distribution| function which has
        been instantiated with parameters.
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

    Creating a fully-connected neural network layer

    .. code-block:: python

        x = Input([0, 1, 2]
        w = Parameter(shape=3)
        b = Parameter()
        out = Relu(Dot(x, w) + b)

    can be accomplished more easily using a :class:`.Dense` layer:

    .. code-block:: python

        x = Input([0, 1, 2]
        out = Dense(x, units=1)

    Especially when the output dimensions are >1 and multiple layers are to be
    stacked:

    .. code-block:: python

        x = Input([0, 1, 2]
        l1 = Dense(x, units=128)
        l2 = Dense(l1, units=64)
        out = Dense(l2, units=1)

    """


    # Layer arguments and their default values
    _default_args = {
        'input': Input(),
    }


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'units': 1,
        'name': 'Dense',
        'activation': tf.nn.relu,
        'weight_posterior': Normal,
        'bias_posterior': Normal,
        'weight_initializer': None,
        'bias_initializer': None,
        'weight_prior': Normal(0, 1),
        'bias_prior': Normal(0, 1),
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['units'], int):
            raise TypeError('units kwarg must be an int')
        if kwargs['units'] < 1:
            raise ValueError('units kwarg must be positive')
        if not isinstance(kwargs['name'], str):
            raise TypeError('name kwarg must be a str')
        if not callable(kwargs['activation']):
            raise TypeError('activation must be a callable')
        if not issubclass(kwargs['weight_posterior'], BaseDistribution):
            raise TypeError('weight_posterior kwarg must be a Distribution')
        if not issubclass(kwargs['bias_posterior'], BaseDistribution):
            raise TypeError('bias_posterior kwarg must be a Distribution')
        _validate_initializer(kwargs['weight_initializer'])
        _validate_initializer(kwargs['bias_initializer'])
        if not isinstance(kwargs['weight_prior'], BaseDistribution):
            raise TypeError('weight_prior kwarg must be a Distribution')
        if not isinstance(kwargs['bias_prior'], BaseDistribution):
            raise TypeError('bias_prior kwarg must be a Distribution')


    def _build(self, args, data, batch_shape):
        """Build the layer."""

        # Inputs
        ndims = args['input'].shape[1].value
        units = self.kwargs['units']
        x_in = tf.reshape(args['input'], batch_shape+[ndims, 1])

        # Create weight and bias parameters
        weight = Parameter(shape=[ndims, units],
                           name=self.kwargs['name']+'_weight',
                           posterior=self.kwargs['weight_posterior'],
                           initializer=self.kwargs['weight_initializer'],
                           prior=self.kwargs['weight_prior'])
        bias = Parameter(shape=[1, units],
                         name=self.kwargs['name']+'_bias',
                         posterior=self.kwargs['bias_posterior'],
                         initializer=self.kwargs['bias_initializer'],
                         prior=self.kwargs['bias_prior'])

        # Build the weight and bias parameter
        weight._build_recursively(data, batch_shape)
        bias._build_recursively(data, batch_shape)

        # Compute output using a sample from the variational posteriors
        weight_samples = weight.built_obj
        bias_samples = tf.reshape(bias.built_obj, batch_shape+[units])
        # TODO: uh, test that this is correct...
        y_out = tf.reduce_sum(weight_samples*x_in, axis=1) + bias_samples
        self._sample = self.kwargs['activation'](y_out)

        # Compute the output using the means of the variational posteriors
        weight_means = weight.mean_obj,
        bias_means = tf.reshape(bias.mean_obj, [1, units])
        mean_y_out = tf.reduce_sum(weight_means*x_in, axis=1) + bias_means
        self._mean = self.kwargs['activation'](mean_y_out)

        # Compute the losses
        self._log_loss_sum = weight._log_loss + bias._log_loss
        self._mean_log_loss_sum = (weight._mean_log_loss +
                                   bias._mean_log_loss)
        self._kl_loss_sum = weight._kl_loss + bias._kl_loss

        # Store weight and bias parameters as args
        self.args['weight'] = weight
        self.args['bias'] = bias

        # Return the sample
        return self._sample


    def _build_mean(self, args, data, batch_shape):
        """Build the layer with mean parameters."""
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

    Where :math:`\mu_i` is the mean of the :math:`i`-th element across the
    batch:

    .. math::

        \mu_i = \frac{1}{N} \sum_{k=1}^{N} x_{ik}

    and :math:`\sigma_i` is the standard deviation of the :math:`i`-th 
    element across the batch:

    .. math::

        \sigma_i = \frac{1}{N} \sum_{k=1}^{N} (x_{ik} - \mu_i)^2

    and :math:`\gamma` and :math:`\beta` are two free parameters for each 
    element.


    Parameters
    ----------
    input : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Input to batch-normalize.


    Keyword Arguments
    -----------------
    proba : bool
        Whether to use probabilistic (True) or point estimates (False) for the
        weight and bias parameters.
        Default = False.
    name : str
        Name for this layer.
        Default = 'BatchNormalization'
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

    Batch normalize the output of a :class:`.Dense` layer:

    .. code-block:: python

        x = Input()
        l1 = Dense(x, units=128)
        l1_norm = BatchNormalization(layer1)
        l2 = Dense(l1_norm, units=64)
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
        'proba': False,
        'name': 'BatchNormalization',
        'weight_posterior': Normal,
        'bias_posterior': Normal,
        'weight_initializer': None,
        'bias_initializer': None,
        'weight_prior': Normal(0, 1),
        'bias_prior': Normal(0, 1),
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['proba'], bool):
            raise TypeError('proba kwarg must be a bool')
        if not isinstance(kwargs['name'], str):
            raise TypeError('name kwarg must be a str')
        if not issubclass(kwargs['weight_posterior'], BaseDistribution):
            raise TypeError('weight_posterior kwarg must be a Distribution')
        if not issubclass(kwargs['bias_posterior'], BaseDistribution):
            raise TypeError('bias_posterior kwarg must be a Distribution')
        _validate_initializer(kwargs['weight_initializer'])
        _validate_initializer(kwargs['bias_initializer'])
        if not isinstance(kwargs['weight_prior'], BaseDistribution):
            raise TypeError('weight_prior kwarg must be a Distribution')
        if not isinstance(kwargs['bias_prior'], BaseDistribution):
            raise TypeError('bias_prior kwarg must be a Distribution')


    def _build(self, args, data, batch_shape):
        """Build the layer."""

        # Inputs
        x_in = args['input']
        dims = x_in.shape[1:]

        # Normalize input elementwise
        x_mean = tf.reduce_mean(x_in, axis=0, keepdims=True)
        x_std = tf.nn.moments(x_in, axes=[0], keepdims=True)
        x_norm = (x_in-x_mean)/x_std

        # Set the variational posterior used for the bias/scaling parameters
        if self.kwargs['proba']: #use full probability distribution
            weight_posterior = self.kwargs['weight_posterior']
            bias_posterior = self.kwargs['bias_posterior']
        else: #only estimate point values
            weight_posterior = Deterministic
            bias_posterior = Deterministic

        # Create weight and bias parameters
        weight = Parameter(shape=dims,
                           name=self.kwargs['name']+'_weight',
                           posterior=weight_posterior,
                           initializer=self.kwargs['weight_initializer'],
                           prior=self.kwargs['weight_prior'])
        bias = Parameter(shape=dims,
                         name=self.kwargs['name']+'_bias',
                         posterior=bias_posterior,
                         initializer=self.kwargs['bias_initializer'],
                         prior=self.kwargs['bias_prior'])

        # Build the weight and bias parameter
        weight._build_recursively(data, batch_shape)
        bias._build_recursively(data, batch_shape)

        # Compute output using a sample from the variational posteriors
        weight_samples = weight.built_obj
        bias_samples = bias.built_obj
        self._sample = x_norm*weight_samples + bias_samples

        # Compute the output using the means of the variational posteriors
        weight_means = weight.mean_obj
        bias_means = bias.mean_obj
        self._mean = x_norm*weight_means + bias_means

        # Compute the losses
        self._log_loss_sum = weight._log_loss + bias._log_loss
        self._mean_log_loss_sum = (weight._mean_log_loss +
                                   bias._mean_log_loss)
        self._kl_loss_sum = weight._kl_loss + bias._kl_loss

        # Store weight and bias parameters as args
        self.args['weight'] = weight
        self.args['bias'] = bias

        # Return the sample
        return self._sample


    def _build_mean(self, args, data, batch_shape):
        """Build the layer with mean parameters."""
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
    """Apply a list of layers sequentially.


    Parameters
    ----------
    input : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Input to the sequence of layers.


    Keyword Arguments
    -----------------
    layers : list of |Layer| s
        List of layers to apply sequentially.  Each layer can take only one
        input.


    Examples
    --------

    Use :class:`.Sequential` to create a multi-layer dense neural network with
    batch normalization:

    .. code-block:: python

        predictions = Sequential([
            Dense(units=128),
            BatchNormalization(),
            Dense(units=64),
            BatchNormalization(),
            Dense(units=1),
        ])
        noise_std = ScaleParameter()
        model = Normal(predictions, noise_std)

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
        layers[-1]._build_recursively(data, batch_shape)

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
        """Build the layer with mean parameters."""
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



class Gather(BaseLayer):
    """Collects slices from a layer using indexes provided by another layer.


    Parameters
    ----------
    values : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Values to gather.
    indices : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Non-negative integers corresponding to the indexes of ``values`` to 
        get.


    Keyword Arguments
    -----------------
    axis : int
        Axis to gather on: integers in ``indices`` correspond to the indices
        of ``values`` in the ``axis``-th dimension.
    index_dtype : |DType|
        Data type to cast ``indices`` to in order to index the values.
        Default = ``tf.uint32``


    Examples
    --------

    Use :class:`.Gather` to look up values based on an index.  For
    example, a random effect can be modeled using :class:`.Gather`:

    .. code-block:: python

        df = pd.DataFrame()
        df['subject_id'] = [0, 0, 1, 1, 2, 2]
        df['feature'] = [1.2, 1.3, 0.1, -0.7, -0.1, 1.7]
        df['target'] = [0.9, 0.5, -1.2, 0.1, -0.6, 0.4]

        subject_id = Input('subject_id')
        feature = Input('feature')
        random_effect = Parameter(shape=3)
        fixed_effect = Parameter()

        predictions = (Gather(random_effect, subject_id)
                       + fixed_effect*feature)
        model = Normal(predictions, 1.0)

        model.fit(['subject_id', 'feature'], 'target', data=df)

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
            raise TypeError('axis kwarg must be an int')
        if not isinstance(kwargs['index_dtype'], tf.DType):
            raise TypeError('index_dtype kwarg must be a tf.DType')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.gather(args['values'], 
                         tf.cast(args['indexes'], self.kwargs['index_dtype']), 
                         axis=self.kwargs['axis'])



class Embedding(BaseLayer):
    r"""A categorical embedding layer.

    Maps an input variable containing non-negative integers to dense vectors.
    The length of the vectors (the dimensionality of the embedding) can be set
    with the ``dims`` keyword argument.  The embedding is learned over the
    course of training: if there are N unique integers in the input, and the
    embedding dimensionality is M, a matrix of NxM free parameters is created
    and optimized to minimize the loss.

    The embeddings can be non-probabilistic (each integer corresponds to a
    single point in M-dimensional space, the default), or probabilistic (each
    integer corresponds to a M-dimensional multivariate distribution).

    By default, a :class:`.Deterministic` distribution is used for the
    embedding variables' posterior distributions, with :class:`.Normal`
    ``(0, 1)`` priors.  This corresponds to normal non-probabilistic embedding
    with L2 regularization.


    Parameters
    ----------
    input : int, float, |ndarray|, |Tensor|, |Variable|, |Parameter|, or |Layer|
        Input non-negative integers to embed in a ``dims``-dimensional space.


    Keyword Arguments
    -----------------
    dims : int > 0
        Number of embedding dimensions.
    name : str
        Name for this layer.
        Default = 'Embedding'
    proba : bool
        Whether to use probabilistic (True) or point estimates (False) for the
        embeddings.
        Default = False.
    posterior : |Distribution|
        Probability distribution class to use to approximate the embeddings'
        posterior distributions.
        Default = :class:`.Normal`
    prior : |None| or a |Distribution| object
        Prior probability distribution for the embeddings. |None| or a 
        |Distribution| function which has been instantiated with parameters.
        Default = :class:`.Normal` ``(0,1)``
    initializer : {|None| or dict or |Tensor| or |Initializer|}
        Initializer for the embeddings' variational posterior parameters.
        See :class:`.Parameter` for more information on how to specify a
        custom ``initializer``.
    index_dtype : |DType|
        Data type to cast input to in order to look up embedding parameters.
        Default = ``tf.uint32``


    Examples
    --------

    Embed word IDs into a 50-dimensional space:

    .. code-block:: python

        x = Input('word_id')
        emb = Embedding(x, dims=50)
        l1 = Dense(emb, units=128)
        predictions = Dense(l1, units=1)
        ...


    Notes
    -----
    
    With probabilistic embeddings (when ``proba``=``True``), the embeddings
    parameters are created as a matrix of independent parameters.  That is,
    the covariance structure of the embedding posteriors is not modeled.
    A multivariate distribution is also not used because the sum of the KL
    divergence between multiple univariate distributions is equal to the 
    KL divergence between two multivariate distributions with zero covariance
    (with corresponding parameters in each dimension), as is the sum of the 
    log posterior probabilities.  In other words, we use a NxM matrix of 
    independent parameters instead of N M-dimensional parameters.

    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'dims': 5,
        'name': 'Embedding',
        'proba': False,
        'posterior': Normal,
        'initializer': None,
        'prior': Normal(0, 1),
        'index_dtype': tf.uint32,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['dims'], int):
            raise TypeError('dims kwarg must be an int')
        if not isinstance(kwargs['name'], str):
            raise TypeError('name kwarg must be a str')
        if kwargs['dims'] < 1:
            raise ValueError('dims kwarg must be positive')
        if not isinstance(kwargs['proba'], bool):
            raise ValueError('proba kwarg must be a bool')
        if not issubclass(kwargs['posterior'], BaseDistribution):
            raise TypeError('posterior kwarg must be a Distribution class')
        _validate_initializer(kwargs['initializer'])
        if not isinstance(kwargs['prior'], BaseDistribution):
            raise TypeError('prior kwarg must be a Distribution object')
        if not isinstance(kwargs['index_dtype'], tf.DType):
            raise TypeError('index_dtype kwarg must be a tf.DType')


    def _build(self, args, data, batch_shape):
        """Build the layer."""

        # Ensure arg is Input layer w/ only 1 dimension
        if not isinstance(self.args['input'], Input):
            raise TypeError('Embedding input must be an Input layer')
        if isinstance(self.args['input']._int_cols, list):
            raise RuntimeError('Embedding input must be 1-dimensional')

        # Inputs
        x_in = args['input']
        input_dim = self.args['input']._nunique

        # Set the variational posterior used for the embedding parameters
        if self.kwargs['proba']: #use full probability distribution
            posterior = self.kwargs['posterior']
        else: #only estimate point values
            posterior = Deterministic

        # Create embedding parameters
        embeddings = Parameter(shape=[input_dim, self.kwargs['dims']],
                               name=self.kwargs['name'],
                               posterior=posterior,
                               initializer=self.kwargs['initializer'],
                               prior=self.kwargs['prior'])

        # Build the embedding parameters
        embeddings._build_recursively(data, batch_shape)

        # TODO: need to figure out gathering samples
        # embeddings.built_obj is shape [batch_size, input_dim, dims]

        # Compute output using a sample from the variational posteriors
        self._sample = tf.gather(embeddings.built_obj, 
                                 tf.cast(x_in, self.kwargs['index_dtype']))

        # Compute the output using the means of the variational posteriors
        self._mean = tf.gather(embeddings.mean_obj, 
                               tf.cast(x_in, self.kwargs['index_dtype']))

        # Compute the losses
        self._log_loss_sum = embeddings._log_loss
        self._mean_log_loss_sum = embeddings._log_loss
        self._kl_loss_sum = embeddings._kl_loss

        # Store embeddings parameters as an arg
        self.args['embeddings'] = embeddings

        # Return the sample
        return self._sample


    def _build_mean(self, args, data, batch_shape):
        """Build the layer with mean parameters."""
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


# TODO: LSTM
