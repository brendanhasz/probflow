"""Layers.

Layers are objects which take one or more tensors as input arguments, 
perform some computation on them, and output a single tensor.  The input 
tensor(s) can be the input data, constants, or the outputs of other layers.
Layers can additionally take keyword arguments which control how the 
computation is performed.


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

* :class:`.Reshape` - reshape vectors/matrixes
* :class:`.Cat` - concatenate vectors/matrixes
* :class:`.Dot` - dot product
* :class:`.Matmul` - matrix multiplication


Neural Network Layers
---------------------

These layers perform various neural-network-related operations.  Some of them,
including :class:`.Dense`, :class:`.BatchNormalization`, and 
:class:`.Embedding` add new |Parameters| to the model.

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
    'Reshape',
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


def _broadcast2(a, b, op):
    """Attempt to broadcast two |Tensors|"""
    if isinstance(a, tf.Tensor) and isinstance(b, tf.Tensor):
        if len(a.shape) > len(b.shape):
            #return op(a, tf.broadcast_to(b, new_shape))
            return op(a, b[..., tf.newaxis])
        elif len(a.shape) < len(b.shape):
            #return op(tf.broadcast_to(a, b.shape), b)
            return op(a[..., tf.newaxis], b)
        else:
            return op(a, b)
    else:
        return op(a, b)



class Input(BaseLayer):
    r"""Layer which represents the input data.

    This layer represents the input data, either all of it, or a subset of
    specific columns.  Which columns to use for this ``Input``` can be 
    specified using the ``cols`` keyword argument.  You can use integers
    to specify which column of the input matrix to use if the input 
    data is an |ndarray|.  Or, you can use strings or a list of strings
    when your input data is a |DataFrame|.  See the examples for more.


    Keyword Arguments
    -----------------
    cols : None or int or str or list of int or str
        Columns of the independent variables matrix to use.


    Examples
    --------

    To create an object which corresponds to *all* the columns of the input
    data, call :class:`.Input` with no arguments::

        from probflow import Input

        features = Input()

    To use one specific column from the data matrix, use the ``cols``
    keyword argument::

        x0 = Input(cols=0)
        x1 = Input(cols=1)

    Then you can use these subsets of the data as input to different parts of
    the model::

        w0 = Parameter() #weight for data in 1st col
        w1 = Parameter() #weight for data in 2nd col
        b = Parameter() #bias

        predictions = w0*x0 + w1*x1 + b
        model = Normal(predictions, 1.0)

    When you fit the model, the data from the appropriate columns will be 
    piped to the appropriate places::

        # generate dummy data
        X = np.random.randn(1000, 2)
        w = np.array([[-0.3], [0.5]])
        y = np.sum(X*w, axis=1) + 1.0 + np.random.randn(1000)

        model.fit(X, y)
        #x0 contains data in X[:, 0]
        #x1 contains data in X[:, 1]

    To use multiple columns in a single ``Input`` object, set ``cols`` to be
    a list of integers::

        x_lin = Input(cols=[0, 1])
        x_exp = Input(cols=2)

    Then the data in that ``Input`` object is multidimensional::

        w_lin = Parameter(shape=2)
        w_exp = Parameter()

        predictions = Dot(w_lin, x_lin) + Exp(w_exp*x_exp)
        model = Normal(predictions, 1.0)

    ``Input`` layers can also take string arguments to make working with 
    |DataFrames| easier.  Supposing we have a DataFrame with information about
    housing prices::

        import pandas as pd
        df = pd.DataFrame()
        df['price'] = [0.1, 0.2, 0.3, 0.5] #(in millions)
        df['sq_ft'] = [1500, 2000, 2500, 3000]
        df['floors'] = [2, 2, 1, 3]
        df['state'] = ['VT', 'MA', 'MN', 'WA']

    When passing the dependent variable data as a |DataFrame|, you can use 
    a string to specify what column this ``Input`` should correspond to::

        x_sq_ft = Input(cols='sq_ft')
        x_floors = Input(cols='floors')

        w_sq_ft = Parameter()
        w_floors = Parameter()
        baseline = Parameter()

        predictions = w_sq_ft*x_sq_ft + w_floors*x_floors + baseline
        model = Normal(predictions, 1.0)

    Then when fitting the model, you can pass ``x`` as a |DataFrame| and 
    ``y`` as a |Series|::

        model.fit(df[['sq_ft', 'floors']], df['price'])

    Multiple columns can also be specified for a single input using a list of
    strings::

        x_cont = Input(cols=['sq_ft', 'floors'])
        x_cat = Input(cols='state')

        w_cont = Parameter(shape=2)
        state_effect = Embedding(x_cat, dims=1)

        predictions = Dot(w_cont, x_cont) + state_effect
        model = Normal(predictions, 1.0)

    With this method you must again pass ``x`` and ``y`` as |DataFrames| or
    |Series|.
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
    r"""A layer which adds two inputs, elementwise.

    Given :math:`a` and :math:`b`, this layer returns 
    :math:`a+b`, elementwise.


    Examples
    --------

    Use the ``Add`` layer to add two inputs, elementwise::

        from probflow import Input, Add, Parameter, Exp

        x0 = Input(0)
        x1 = Input(1)
        sum1 = Add(x0, x1) # x0+x1

    The ``Add`` layer can also take |Parameters| as input::

        w1 = Parameter()
        sum2 = Add(sum1, w1)

    As well as |Tensors|::

        sum3 = Add(tf.constant(1.0), sum2)

    or other |Layers|::

        sum4 = Add(sum3, Exp(sum3))

    ProbFlow objects define the ``__add__`` special method to use this layer.
    That is::

        x0+x1

    is equivalent to::

        Add(x0, x1)
    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        op = lambda a, b: a + b
        return _broadcast2(args['a'], args['b'], op)



class Sub(BaseLayer):
    r"""A layer which subtracts one input from another, elementwise.

    Given :math:`a` and :math:`b`, this layer returns 
    :math:`a-b`, elementwise.


    Examples
    --------

    Use the ``Sub`` layer to subtract one input from another, elementwise::

        from probflow import Input, Sub, Parameter, Exp

        x0 = Input(0)
        x1 = Input(1)
        diff1 = Sub(x0, x1) # x0 - x1

    The ``Sub`` layer can also take |Parameters| as input::

        w1 = Parameter()
        diff2 = Sub(diff1, w1)

    As well as |Tensors|::

        diff3 = Sub(tf.constant(1.0), diff2)

    or other |Layers|::

        diff4 = Sub(diff3, Exp(diff3))

    ProbFlow objects define the ``__sub__`` special method to use this layer.
    That is::

        x0-x1

    is equivalent to::

        Sub(x0, x1)
    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        op = lambda a, b: a - b
        return _broadcast2(args['a'], args['b'], op)



class Mul(BaseLayer):
    r"""A layer which multiplies two inputs, elementwise.

    Given :math:`a` and :math:`b`, this layer returns 
    :math:`a*b`, elementwise.


    Examples
    --------

    Use the ``Mul`` layer to multiply two inputs, elementwise::

        from probflow import Input, Mul, Parameter, Exp

        x0 = Input(0)
        x1 = Input(1)
        prod1 = Mul(x0, x1) # x0*x1

    The ``Mul`` layer can also take |Parameters| as input::

        w1 = Parameter()
        prod2 = Mul(prod1, w1)

    As well as |Tensors|::

        prod3 = Mul(tf.constant(1.0), prod2)

    or other |Layers|::

        prod4 = Mul(prod3, Exp(prod3))

    ProbFlow objects define the ``__mul__`` special method to use this layer.
    That is::

        x0*x1

    is equivalent to::

        Mul(x0, x1)
    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        op = lambda a, b: a * b
        return _broadcast2(args['a'], args['b'], op)



class Div(BaseLayer):
    r"""A layer which divides one input by another, elementwise.

    Given :math:`a` and :math:`b`, this layer returns 
    :math:`a/b`, elementwise.


    Examples
    --------

    Use the ``Div`` layer to multiply two inputs, elementwise::

        from probflow import Input, Div, Parameter, Exp

        x0 = Input(0)
        x1 = Input(1)
        ratio1 = Div(x0, x1) # x0/x1

    The ``Div`` layer can also take |Parameters| as input::

        w1 = Parameter()
        ratio2 = Div(ratio1, w1)

    As well as |Tensors|::

        prod3 = Div(tf.constant(1.0), ratio2)

    or other |Layers|::

        ratio4 = Div(ratio3, Exp(ratio3))

    ProbFlow objects define the ``__div__`` special method to use this layer.
    That is::

        x0/x1

    is equivalent to::

        Div(x0, x1)
    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        op = lambda a, b: a / b
        return _broadcast2(args['a'], args['b'], op)



class Neg(BaseLayer):
    r"""A layer which outputs the negative of its input.

    Given :math:`x`, this layer returns :math:`-x`, elementwise.

    .. image:: img/layers/neg.svg
        :width: 50 %
        :align: center


    Examples
    --------

    Use the ``Neg`` layer to negate an input, elementwise::

        from probflow import Input, Neg, Parameter, Exp

        x0 = Input(0)
        neg1 = Neg(x0)

    The ``Neg`` layer can also take |Parameters| as input::

        w1 = Parameter()
        neg2 = Neg(w1)

    As well as |Tensors|::

        neg3 = Neg(tf.constant(1.0))

    or other |Layers|::

        neg4 = Neg(neg1+neg2)

    ProbFlow objects define the ``__neg__`` special method to use this layer.
    That is::

        -x0

    is equivalent to::

        Neg(x0)
    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return -args['input']



class Abs(BaseLayer):
    r"""A layer which outputs the absolute value of its input.

    Given :math:`x`, this layer returns :math:`|x|`, elementwise.

    .. image:: img/layers/abs.svg
        :width: 50 %
        :align: center    


    Examples
    --------

    Use the ``Abs`` layer to take the absolute value of an input, 
    elementwise::

        from probflow import Input, Abs, Parameter, Exp

        x0 = Input(0)
        abs1 = Abs(x0)

    The ``Abs`` layer can also take |Parameters| as input::

        w1 = Parameter()
        abs2 = Abs(w1)

    As well as |Tensors|::

        abs3 = Abs(tf.constant(1.0))

    or other |Layers|::

        abs4 = Abs(Sub(abs1, abs2))

    ProbFlow objects define the ``__abs__`` special method to use this layer.
    That is::

        abs(x0)

    is equivalent to::

        Abs(x0)
    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return abs(args['input'])



class Exp(BaseLayer):
    r"""A layer which outputs the natural exponent of its input.

    Given :math:`x`, this layer returns :math:`e^x`, elementwise.

    .. image:: img/layers/exp.svg
        :width: 50 %
        :align: center


    Examples
    --------

    Use the ``Exp`` layer to compute :math:`e` to the power of some data::

        from probflow import Input, Exp, Parameter

        x0 = Input(0)
        exp1 = Exp(x0)

    The ``Exp`` layer can also take |Parameters| as input::

        w1 = Parameter()
        exp2 = Exp(w1)

    As well as |Tensors|::

        exp3 = Exp(tf.constant(1.0))

    or other |Layers|::

        exp4 = Exp(Sub(exp1, exp2))
    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.exp(args['input'])



class Log(BaseLayer):
    r"""A layer which outputs the natural log of its input.

    Given :math:`x`, this layer returns :math:`\ln x`, elementwise.

    .. image:: img/layers/log.svg
        :width: 50 %
        :align: center


    Examples
    --------

    Use the ``Log`` layer to compute the natural log of some data::

        from probflow import Input, Log, Parameter

        x0 = Input(0)
        log1 = Log(x0)

    The ``Log`` layer can also take |Parameters| as input::

        w1 = Parameter()
        log2 = Log(w1)

    As well as |Tensors|::

        log3 = Log(tf.constant(1.0))

    or other |Layers|::

        log4 = Log(Add(log1, log2))
    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.log(args['input'])



class Reciprocal(BaseLayer):
    r"""A layer which outputs the reciprocal of its input.

    Given :math:`x`, this layer returns (elementwise):

    .. math::

        \text{Reciprocal}(x) = \frac{1}{x}

    .. image:: img/layers/reciprocal.svg
        :width: 50 %
        :align: center


    Examples
    --------

    Use the ``Reciprocal`` layer to compute the inverse of some data::

        from probflow import Input, Reciprocal, Parameter

        x0 = Input(0)
        inv1 = Reciprocal(x0)

    The ``Reciprocal`` layer can also take |Parameters| as input::

        w1 = Parameter()
        inv2 = Reciprocal(w1)

    As well as |Tensors|::

        inv3 = Reciprocal(tf.constant(1.0))

    or other |Layers|::

        inv4 = Reciprocal(Add(inv1, inv2))
    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reciprocal(args['input'])



class Sqrt(BaseLayer):
    r"""A layer which outputs the square root of its input.

    Given :math:`x`, this layer returns (elementwise):

    .. math::

        \text{Sqrt}(x) = \sqrt{x}

    .. image:: img/layers/sqrt.svg
        :width: 50 %
        :align: center


    Examples
    --------

    Use the ``Sqrt`` layer to compute the square root of some data::

        from probflow import Input, Sqrt, Parameter

        x0 = Input(0)
        sqrt1 = Sqrt(x0)

    The ``Sqrt`` layer can also take |Parameters| as input::

        w1 = Parameter()
        sqrt2 = Sqrt(w1)

    As well as |Tensors|::

        sqrt3 = Sqrt(tf.constant(1.0))

    or other |Layers|::

        sqrt4 = Sqrt(Add(sqrt1, sqrt2))
    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.sqrt(args['input'])



class Transform(BaseLayer):
    r"""Performs an elementwise transform using arbitrairy |TensorFlow| ops. 

    Given :math:`x`, and some function :math:`f`, this layer returns 
    :math:`f(x)`, elementwise.


    Keyword Arguments
    -----------------
    func : callable
        Funtion to use to transform the input data.  Should use only 
        |TensorFlow| ops.


    Examples
    --------

    Use the ``Transform`` layer to implement custom transformation not 
    implemented in ProbFlow.  For example, to create a layer which takes the
    floor of its input (rounds down to the next lowest integer, a transform
    which is not included in ProbFlow), use the ``Transform`` layer::

        import tensorflow as tf
        from probflow import Input, Transform

        x0 = Input(0)
        floored = Transform(x0, func=lambda x: tf.floor(x))

    Or, to create a layer which computes the :math:`sin` of the input::

        sinx = Transform(x0, func=lambda x: tf.sin(x))
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

    Given :math:`x`, this layer returns (elementwise):

    .. math::

        \text{Sigmoid}(x) = \frac{1}{1 + \exp (-x)}

    .. image:: img/layers/sigmoid.svg
        :width: 50 %
        :align: center


    Examples
    --------

    Use the ``Sigmoid`` layer to pass some data through a sigmoid::

        from probflow import Input, Dense, Sigmoid, Bernoulli

        x0 = Input(0)
        logits = Dense(x0)
        probs = Sigmoid(logits)
        model = Bernoulli(probs, input_type='probs')

    The ``Sigmoid`` layer can also take |Parameters| as input::

        w1 = Parameter()
        sig1 = Sigmoid(w1)

    As well as |Tensors|::

        sig2 = Sigmoid(tf.constant(1.0))

    or other |Layers|::

        sig3 = Sigmoid(Add(1.0, 2.0))
    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.sigmoid(args['input'])



class Relu(BaseLayer):
    r"""A layer which linearly rectifies its input, elementwise.

    Given :math:`x`, this layer returns (elementwise):

    .. math::

        \text{Relu}(x) = \max (x, 0)

    .. image:: img/layers/relu.svg
        :width: 50 %
        :align: center


    Examples
    --------

    Use the ``Relu`` layer to linearly rectify some input::

        from probflow import Input, Dense, Sigmoid, Bernoulli

        x0 = Input()
        activations = Relu(Dense(x0))

    The ``Relu`` layer can also take |Parameters| as input::

        w1 = Parameter()
        relu1 = Relu(w1)

    As well as |Tensors|::

        relu2 = Relu(tf.constant(1.0))

    or other |Layers|::

        relu3 = Relu(Add(1.0, 2.0))
    """

    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.nn.relu(args['input'])



class Softmax(BaseLayer):
    r"""A layer which outputs the softmax of its input.

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{Softmax}(\mathbf{x}) = \mathbf{\sigma}

    where

    .. math::

        \sigma_i = \frac{\exp (x_i)}{\sum_j \exp (x_j)}

    The default is to compute the softmax along the the last dimension of the 
    input |Tensor|, but this can be set with the ``axis`` keyword argument.
    The output has the same shape as the input.
    

    Keyword Arguments
    -----------------
    axis : int
        What axis to compute the operation along.  
        Default is -1 (the last dimension).


    Examples
    --------

    Use the ``Softmax`` layer to normalize vector(s) such that elements of the
    vector sum to 1.  For example, to convert the output of a fully-connected
    layer to class probabilities::

        from probflow import Input, Softmax, Bernoulli

        x0 = Input()
        raw_vals = Dense(x0, units=10)
        probs = Softmax(raw_vals)
        model = Bernoulli(probs, input_type='probs')

    To compute the softmax over a specific dimension, use the ``axis`` keyword
    argument.  For example, to perform the softmax along the *second*-to-last 
    dimension::

        probs = Softmax(raw_vals, axis=-2)
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

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{Sum}(\mathbf{x}) = \sum_i x_i

    The default is to compute the sum along the the last dimension of the 
    input |Tensor|, but this can be set with the ``axis`` keyword argument.
    The output is *not* the same shape as the input - that is, this is a
    reduction layer.


    Keyword Arguments
    -----------------
    axis : int
        What axis to compute the operation along.  
        Default is -1 (the last dimension).
    keepdims : bool
        Whether the output should keep the dimensions of the input (but with
        a size in the ``axis`` dimension of 1) or reduce the dimensions.


    Examples
    --------

    Use ``Sum`` to sum elements of a vector.  For example, when doing a linear
    regression, we sum the weight-feature products::

        from probflow import Input, Parameter, Sum

        features = Input([0, 1, 2, 3, 4])
        weights = Parameter(shape=5)
        bias = Parameter()

        predictions = Sum(weights*features) + bias

    This layer sums along the last dimension by default::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        summed = Sum(vals)
        # summed has shape (2, 3)

    However this can be changed by setting the ``axis`` keyword argument::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        summed = Sum(vals, axis=-2)
        # summed has shape (2, 4)
    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
        'keepdims': True,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_sum(args['input'], 
                             axis=self.kwargs['axis'], 
                             keepdims=self.kwargs['keepdims'])



class Mean(BaseLayer):
    r"""A layer which outputs the mean of its inputs.

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{Mean}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^N x_i

    The default is to compute the mean along the the last dimension of the 
    input |Tensor|, but this can be set with the ``axis`` keyword argument.
    The output is *not* the same shape as the input - that is, this is a
    reduction layer.


    Keyword Arguments
    -----------------
    axis : int
        What axis to compute the operation along.  
        Default is -1 (the last dimension).
    keepdims : bool
        Whether the output should keep the dimensions of the input (but with
        a size in the ``axis`` dimension of 1) or reduce the dimensions.


    Examples
    --------

    Use ``Mean`` to take the average of elements in a vector.  By default,
    this layer takes the average along the last dimension::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        average = Mean(vals)
        # average has shape (2, 3)

    However this can be changed by setting the ``axis`` keyword argument::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        summed = Mean(vals, axis=-2)
        # summed has shape (2, 4)
    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
        'keepdims': True,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_mean(args['input'], 
                              axis=self.kwargs['axis'],
                              keepdims=self.kwargs['keepdims'])



class Min(BaseLayer):
    r"""A layer which outputs the minimum of its inputs.

    Given a vector :math:`\mathbf{x}`, this layer returns 
    :math:`\min \mathbf{x}` along one dimension.  
    The dimensionality of the output of this layer is less than the 
    dimensionality of the input.

    The default is to compute the minimum along the the last dimension of the 
    input |Tensor|, but this can be set with the ``axis`` keyword argument.
    The output is *not* the same shape as the input - that is, this is a
    reduction layer.


    Keyword Arguments
    -----------------
    axis : int
        What axis to compute the operation along.  
        Default is -1 (the last dimension).
    keepdims : bool
        Whether the output should keep the dimensions of the input (but with
        a size in the ``axis`` dimension of 1) or reduce the dimensions.


    Examples
    --------

    Use ``Min`` to take the minimum of elements in a vector.  By default,
    this layer takes the minimum along the last dimension::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        minimum = Min(vals)
        # minimum has shape (2, 3)

    However this can be changed by setting the ``axis`` keyword argument::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        minimum = Min(vals)
        # minimum has shape (2, 4)
    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
        'keepdims': True,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_min(args['input'], 
                             axis=self.kwargs['axis'],
                             keepdims=self.kwargs['keepdims'])



class Max(BaseLayer):
    r"""A layer which outputs the maximum of its inputs.

    Given a vector :math:`\mathbf{x}`, this layer returns 
    :math:`\max \mathbf{x}` along one dimension.  
    The dimensionality of the output of this layer is less than the 
    dimensionality of the input.

    The default is to compute the maximum along the the last dimension of the 
    input |Tensor|, but this can be set with the ``axis`` keyword argument.
    The output is *not* the same shape as the input - that is, this is a
    reduction layer.


    Keyword Arguments
    -----------------
    axis : int
        What axis to compute the operation along.  
        Default is -1 (the last dimension).
    keepdims : bool
        Whether the output should keep the dimensions of the input (but with
        a size in the ``axis`` dimension of 1) or reduce the dimensions.


    Examples
    --------

    Use ``Max`` to take the maximum of elements in a vector.  By default,
    this layer takes the maximum along the last dimension::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        maximum = Max(vals)
        # maximum has shape (2, 3)

    However this can be changed by setting the ``axis`` keyword argument::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        maximum = Max(vals)
        # maximum has shape (2, 4)
    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
        'keepdims': True,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_max(args['input'], 
                             axis=self.kwargs['axis'],
                             keepdims=self.kwargs['keepdims'])



class Prod(BaseLayer):
    r"""A layer which outputs the product of its inputs.

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{Sum}(\mathbf{x}) = \prod_i x_i

    The dimensionality of the output of this layer is less than the 
    dimensionality of the input.

    The default is to compute the product along the the last dimension of the 
    input |Tensor|, but this can be set with the ``axis`` keyword argument.
    The output is *not* the same shape as the input - that is, this is a
    reduction layer.


    Keyword Arguments
    -----------------
    axis : int
        What axis to compute the operation along.  
        Default is -1 (the last dimension).
    keepdims : bool
        Whether the output should keep the dimensions of the input (but with
        a size in the ``axis`` dimension of 1) or reduce the dimensions.


    Examples
    --------

    Use ``Prod`` to take the product of elements in a vector.  By default,
    this layer takes the product along the last dimension::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        product = Prod(vals)
        # product has shape (2, 3)

    However this can be changed by setting the ``axis`` keyword argument::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        product = Prod(vals)
        # product has shape (2, 4)
    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
        'keepdims': True,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_prod(args['input'], 
                              axis=self.kwargs['axis'],
                              keepdims=self.kwargs['keepdims'])



class LogSumExp(BaseLayer):
    r"""A layer which outputs log(sum(exp(inputs))).

    Given a vector :math:`\mathbf{x}`, this layer returns:

    .. math::

        \text{LogSumExp}(\mathbf{x}) = \log \left( \sum_i \exp x_i \right)

    but using a method robust to underflow when you're working with values
    in log-space.  Specifically, it first computes the maximum of the vector

    .. math::

        m = \max \mathbf{x}

    and then computes the natural exponent of the values after subtracting the
    max, sums the result, computes the natural log of the result, and then 
    adds the max back in logspace:

    .. math::

        \text{LogSumExp}(\mathbf{x}) = 
        \log \left( \sum_i \exp (x_i-m) \right) + m

    The default is to compute the operation along the the last dimension of 
    the input |Tensor|, but this can be set with the ``axis`` keyword
    argument. The output is *not* the same shape as the input - that is, this
    is a reduction layer.
    

    Keyword Arguments
    -----------------
    axis : int
        What axis to compute the operation along.  
        Default is -1 (the last dimension).
    keepdims : bool
        Whether the output should keep the dimensions of the input (but with
        a size in the ``axis`` dimension of 1) or reduce the dimensions.


    Examples
    --------

    Use ``LogSumExp`` to the sum of elements in a vector in non-log space when
    your values are in log-space, in a numerically robust way.  The default is
    to perform the operation along the last dimension::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        out = LogSumExp(vals)
        # out has shape (2, 3)

    However this can be changed by setting the ``axis`` keyword argument::

        vals = tf.random.normal([2, 3, 4])
        # vals has shape (2, 3, 4)
        out = LogSumExp(vals, axis=-2)
        # out has shape (2, 4)
    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
        'keepdims': True,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_logsumexp(args['input'], 
                                   axis=self.kwargs['axis'],
                                   keepdims=self.kwargs['keepdims'])



class Reshape(BaseLayer):
    r"""A layer which reshapes its input.

    Reshapes a tensor.  The default is to flatten the tensor (make it 1D).
    

    Keyword Arguments
    -----------------
    shape : list of int
        What the new shape of the tensor should be.
        Use -1 to force all remaining dimensions into one dimension.


    Examples
    --------

    Use the ``Reshape`` layer to change the shape of a tensor::

        from probflow import Parameter, Reshape

        weights = Parameter(shape=[4, 3, 2])
        reshaped_weights = Reshape(weights, shape=[4, 6])

    Use a shape of -1 to force all remaining dimensions into one dimension::
        
        weights = Parameter(shape=[4, 3, 2, 1])
        reshaped_weights = Reshape(weights, shape=[4, -1])
        # reshaped_weights has shape (4, 6)
    """


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'shape': None,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if (kwargs['shape'] is not None and
                not isinstance(kwargs['shape'], (list, int))):
            raise ValueError('shape kwarg must be a list or an int')
        if isinstance(kwargs['shape'], list):
            for e in kwargs['shape']:
                if not isinstance(e, int):
                    raise TypeError('each element of shape kwarg must be int')


    def _build(self, args, _data, batch_shape):
        """Build the layer."""
        if isinstance(self.kwargs['shape'], list):
            new_shape = tf.concat([batch_shape, self.kwargs['shape']], axis=0)
            return tf.reshape(args['input'], new_shape)
        elif isinstance(self.kwargs['shape'], int):
            new_shape = tf.concat([batch_shape, [self.kwargs['shape']]], 
                                  axis=0)
            return tf.reshape(args['input'], new_shape)
        else:
            new_shape = tf.concat([batch_shape, [-1]], axis=0)
            return tf.reshape(args['input'], new_shape)



class Cat(BaseLayer):
    r"""A layer which concatenates its two inputs.

    Given two tensors, whose shapes must be the same in every dimension except
    the ``axis``-th dimension (where ``axis`` is a keyword argument), this
    layer concatenates the two tensors along the ``axis``-th dimension.
    The default is to concatenate along the last dimension.

    TODO: really we want to be able to pass a LIST of inputs, not just 2...


    Keyword Arguments
    -----------------
    axis : int
        What axis to compute the operation along.  
        Default is -1 (the last dimension).


    Examples
    --------

    Use ``Cat`` to concatenate two tensors.  By default, ``Cat`` concatenates
    on the last dimension::

        a = tf.random.normal([2, 3, 4])
        b = tf.random.normal([2, 3, 5])
        concat = Cat(a, b)
        # concat has shape (2, 3, 9)

    However, this can be changed using the ``axis`` keyword argument::

        a = tf.random.normal([2, 4, 3])
        b = tf.random.normal([2, 5, 3])
        concat = Cat(a, b, axis=-2)
        # concat has shape (2, 9, 3)

    Note that the dimensions of both tensors must be equal for all dimensions
    except the ``axis``-th dimension::

        a = tf.random.normal([2, 3, 4])
        b = tf.random.normal([2, 6, 5])
        concat = Cat(a, b, axis=-1)
        # ERROR!
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

    Given a two vectors :math:`\mathbf{a}` and :math:`\mathbf{b}`,
    this layer returns:

    .. math::

        \text{Dot}(\mathbf{a},\mathbf{b}) =
        \mathbf{a} \cdot \mathbf{b} =
        \sum_i ( a_i b_i )

    The default is to compute the dot product along the the last dimension of 
    the input |Tensor|, but this can be set with the ``axis1`` keyword
    argument (to set the axis of the first input), and the ``axis2`` keyword
    argument (to set the axis of the second input). The output is *not* the
    same shape as the input - that is, this is a reduction layer.


    Parameters
    ----------
    a : float, |ndarray|, |Tensor|, |Layer|, or |Parameter|
        First input
    b : float, |ndarray|, |Tensor|, |Layer|, or |Parameter|
        Second input


    Keyword Arguments
    -----------------
    axis1 : int
        What axis to compute the operation along for the first input ``a``. 
        Default is -1 (the last dimension).
    axis2 : int
        What axis to compute the operation along for the second input ``b``. 
        Default is -1 (the last dimension).


    Examples
    --------

    Use ``Dot`` to take the dot product of two vectors.  For example, in a 
    multiple linear regression, take the dot product of the weights and the
    features::

        from probflow import Input, Parameter, Dot

        features = Input([0, 1, 2, 3, 4])
        weights = Parameter(shape=5)
        bias = Parameter()

        predictions = Dot(features, weights) + bias
    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    # Layer keyword arguments and their default values
    _default_kwargs = {
        'axis': -1,
        'keepdims': True,
    }


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        if not isinstance(kwargs['axis'], int):
            raise ValueError('axis kwarg must be an int')
        if not isinstance(kwargs['keepdims'], bool):
            raise ValueError('keepdims kwarg must be an bool')


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        return tf.reduce_sum(args['a'] * args['b'],
                             axis=self.kwargs['axis'],
                             keepdims=self.kwargs['keepdims'])



class Matmul(BaseLayer):
    r"""A layer which outputs the matrix multiplication of its two inputs.

    Given two matrixes :math:`\mathbf{A}` of shape (N,M) and 
    :math:`\mathbf{B}` of shape (M,P), this layer returns the matrix
    multiplication of the two, a matrix :math:`\mathbf{C}`, of shape (N,P):

    .. math::

        \text{Matmul}(\mathbf{A},\mathbf{B}) =
        \mathbf{A} \mathbf{B} = \mathbf{C}

    where 

    .. math::

        \mathbf{C}_{i,j} = \sum_{k=1}^M \mathbf{A}_{i,k} \mathbf{B}_{k,j}

    .. image:: img/layers/matmul.svg
        :width: 50 %
        :align: center

    Note that this layer can be applied using the matrix multiplication infix
    operator (``@``).  That is, if one has two matrixes::

        A = Parameter(shape=[4, 2])
        B = Parameter(shape=[2, 3])

    Then::

        C = Matmul(A, B)

    is equivalent to::

        C = A @ B


    Parameters
    ----------
    a : |ndarray|, |Tensor|, |Layer|, or |Parameter|
        First input
    b : |ndarray|, |Tensor|, |Layer|, or |Parameter|
        Second input


    Examples
    --------

    Use ``Matmul`` to matrix-multiply two matrices.  For example, in a 
    fully-connected neural network, multiply the input feature vectors by a 
    weight matrix to produce the raw activation values::

        from probflow import Input, Reshape, Parameter, Matmul, Relu

        # Input (D dimensions)
        D = 3
        features = Reshape(Input(), shape=[D, 1])

        # First layer
        weights1 = Parameter(shape=[128, D])
        bias1 = Parameter(shape=128)
        layer1 = Relu(Matmul(weights1, features) + bias1)
    """


    # Layer arguments and their default values
    _default_args = OrderedDict([
        ('a', REQUIRED),
        ('b', REQUIRED)
    ])


    def _build(self, args, _data, _batch_shape):
        """Build the layer."""
        # NOTE: tf.matmul only supports broadcasting as of Apr 2019 nightly!
        op = lambda a, b: tf.reduce_sum(a[..., tf.newaxis] *
                                        b[..., tf.newaxis, :, :],
                                        axis=-2)
        return _broadcast2(args['a'], args['b'], op)



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

        x = Input([0, 1, 2])
        w = Parameter(shape=3)
        b = Parameter()
        out = Relu(Dot(x, w) + b)

    can be accomplished more easily using a :class:`.Dense` layer:

    .. code-block:: python

        x = Input([0, 1, 2])
        out = Dense(x, units=1)

    Especially when the output dimensions are >1 and multiple layers are to be
    stacked:

    .. code-block:: python

        x = Input([0, 1, 2])
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
        if (kwargs['activation'] is not None and 
            not callable(kwargs['activation'])):
            raise TypeError('activation must be a callable or None')
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
        x_shape = tf.concat([batch_shape, [ndims], [1]], axis=0)
        x_in = tf.reshape(args['input'], x_shape)

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
        bias_samples_shape = tf.concat([batch_shape, [units]], axis=0)
        bias_samples = tf.reshape(bias.built_obj, bias_samples_shape)
        y_out = tf.reduce_sum(weight_samples*x_in, axis=1) + bias_samples
        if self.kwargs['activation'] is None:
            self._sample = y_out
        else:
            self._sample = self.kwargs['activation'](y_out)

        # Compute the output using the means of the variational posteriors
        weight_means = weight.mean_obj
        bias_means_shape = tf.concat([[1], [units]], axis=0)
        bias_means = tf.reshape(bias.mean_obj, bias_means_shape)
        mean_y_out = tf.reduce_sum(weight_means*x_in, axis=1) + bias_means
        if self.kwargs['activation'] is None:
            self._mean = mean_y_out
        else:
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
        Default = :class:`.Deterministic`
    bias_posterior : |Distribution|
        Probability distribution class to use to approximate the posterior
        for the bias parameter(s) (:math:`\beta`).
        Default = :class:`.Deterministic`
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
    r"""Apply a list of layers sequentially.


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
    r"""Collects slices from a layer using indexes provided by another layer.


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
    
    With probabilistic embeddings (when ``proba=True``), the embeddings
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
    r"""A 1-dimensional convolutional neural network layer.


    TODO: More info...


    """

    # TODO
    pass



class Conv2d(BaseLayer):
    r"""A 2-dimensional convolutional neural network layer.


    TODO: More info...


    """

    # TODO
    pass



# TODO: Pooling layer


# TODO: LSTM
