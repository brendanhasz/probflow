"""Modules.

Modules are objects which encapsulate some computation, and can contain 
|Parameters|.  For example, a neural network layer is a good example of a 
|Module|, since they store parameters, and use those parameters to perform
a computation (the forward pass of the data through the layer).

* :class:`.Module` - abstract base class for all modules
* :class:`.Dense` - fully-connected neural network layer
* :class:`.Sequential` - apply a list of modules sequentially
* :class:`.BatchNormalization` - normalize data per batch
* :class:`.Embedding` - embed categorical data in a lower-dimensional space

----------

"""

__all__ = [
    'Module',
    'Dense',
    'Sequential',
    'BatchNormalization',
    'Embedding',    
]


import probflow.core.ops as O
from probflow.core.base import BaseModule
from probflow.core.base import BaseParameter
from probflow.core.base import BaseDistribution
from probflow.parameters import Parameter
from probflow.utils.initializers import xavier
from probflow.utils.initializers import scale_xavier



class Module(BaseModule):
    r"""Abstract base class for Modules.

    TODO

    """

    @abstractmethod
    def __init__(self, *args):
        pass


    @abstractmethod
    def __call__(self):
        """Perform the forward pass"""
        pass


    def _get_params(self, obj):
        """Recursively get parameters from an object"""
            if isinstance(attrib, BaseParameter):
                return [attrib]
            elif isinstance(attrib, BaseModule):
                return attrib.parameters()
            elif isinstance(attrib, list):
                return self._list_params(attrib)
            elif isinstance(attrib, dict):
                return self._dict_params(attrib)
            else:
                return []


    def _list_params(self, obj):
        """Recursively search for parameters in lists"""
        params = []
        for e in obj:
            params += self._get_params(e)
        return params



    def _dict_params(self, obj):
        """Recursively search for parameters in lists"""
        params = []
        for _, e in obj.items():
            params += self._get_params(e)
        return params


    def parameters(self):
        """Get a list of all parameters contained in this module and 
        sub-modules.

        TODO

        """
        params = []
        for a in dir(self):
            params += self._get_params(getattr(self, a))
        return params


    def kl_loss(self):
        """Compute the sum of the Kullback–Leibler divergences between
        priors and their variational posteriors for all parameters in this
        module and its sub-modules."""
        return O.sum([p.kl_loss for p in self.parameters()])



class Dense(Module):
    """Dense neural network layer.

    TODO

    Parameters
    ----------
    d_in : int
        Number of input dimensions.
    d_out : int
        Number of output dimensions (number of "units").
    bias : bool
        Whether to include a bias
    name : str
        Name of this layer
    """


    def __init__(self, d_in, d_out, bias=True, name='Dense'):

        # Check types
        if not isinstance(d_in, int):
            raise TypeError('d_in must be an int')
        if d_in < 1:
            raise ValueError('d_in must be >0')
        if not isinstance(d_out, int):
            raise TypeError('d_out must be an int')
        if d_out < 1:
            raise ValueError('d_out must be >0')
        if not isinstance(bias, bool):
            raise TypeError('bias must be True or False')

        # Create the parameters
        self.weights = Parameter(shape=[d_in, d_out])
        if bias:
            self.bias = Parameter(shape=[1, d_out])
        else:
            self.bias = lambda x: 0


    def __call__(self, x):
        """Perform the forward pass"""
        return x @ self.weights() + self.bias()



class Sequential(Module):
    """Apply a series of modules or functions sequentially.

    TODO

    Parameters
    ----------
    steps : list of |Modules| or callables
        Steps to apply
    name : str
        Name of this module
    """


    def __init__(self, steps, name='Sequential'):

        # Check types
        if not isinstance(steps, list):
            raise TypeError('steps must be a list')
        if not all(callable(s) for s in steps):
            raise TypeError('steps must be a list of callables')

        # Store the list of steps
        self.steps = steps


    def __call__(self, x):
        """Perform the forward pass"""
        for step in self.steps:
            x = step(x)
        return x



class BatchNormalization(Module):
    r"""A layer which normalizes its inputs.

    Batch normalization is a technique which normalizes, re-scales, and 
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
    shape : int or list of int or |ndarray|
        Shape of the tensor to be batch-normalized.
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
    weight_initializer : dict of callables
        Initializer functions to use for each variable of the variational
        posterior distribution for the weights (:math:`\gamma`).  Keys
        correspond to variable names (arguments to the distribution), and
        values contain functions to initialize those variables given ``shape``
        as the single argument.
    bias_initializer : dict of callables
        Initializer functions to use for each variable of the variational
        posterior distribution for the biases (:math:`\beta`).  Keys
        correspond to variable names (arguments to the distribution), and
        values contain functions to initialize those variables given ``shape``
        as the single argument.


    Examples
    --------

    Batch normalize the output of a :class:`.Dense` layer:

    .. code-block:: python

        import probflow as pf

        network = pf.Sequential([
            pf.Dense(d_in=7, d_out=100, bias=False),
            pf.BatchNormalization(100),
            tf.nn.relu,
            pf.Dense(d_in=100, d_out=1)
        ])
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

    def __init__(self, 
                 shape,
                 weight_posterior=Deterministic,
                 bias_posterior=Deterministic,
                 weight_prior=Normal(0, 1),
                 bias_prior=Normal(0, 1),
                 weight_initializer={'loc': xavier},
                 bias_initializer={'loc': xavier},
                 name='BatchNormalization'):

        # Create the parameters
        self.weight = Parameter(shape=shape,
                                posterior=weight_posterior,
                                prior=weight_prior,
                                initializer=weight_initializer,
                                name=name+'weight')
        self.bias = Parameter(shape=shape,
                              posterior=bias_posterior,
                              prior=bias_prior,
                              initializer=bias_initializer,
                              name=name+'bias')


    def __call__(self, x):
        """Perform the forward pass"""
        return self.weight()*(x-O.mean(x, axis=0))/O.std(x, axis=0)+self.bias()



class Embedding(Module):
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
    k : int > 0
        Number of categories to embed.
    d : int > 0
        Number of embedding dimensions.
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Deterministic`
    prior : |Distribution| object
        Prior probability distribution which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0,1)``
    initializer : dict of callables
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    name : str
        Name for this layer.
        Default = 'Embeddings'


    Examples
    --------

    Embed 10k word IDs into a 50-dimensional space:

    .. code-block:: python

        emb = Embedding(k=10000, d=50)

        ids = tf.random.uniform([1000000], minval=1, maxval=10000, 
                                dtype=tf.dtypes.int64)

        embeddings = emb(ids)

    TODO: fuller example

    """

    def __init__(self, 
                 k,
                 d,
                 posterior=Deterministic,
                 prior=Normal(0, 1),
                 initializer={'loc': xavier},
                 name='Embeddings'):

        # Check types
        if not isinstance(k, int):
            raise TypeError('k must be an int')
        if k < 1:
            raise ValueError('k must be >0')
        if not isinstance(d, int):
            raise TypeError('d must be an int')
        if d < 1:
            raise ValueError('d must be >0')

        # Create the parameters
        self.embeddings = Parameter(shape=[k, d],
                                    posterior=posterior,
                                    prior=prior,
                                    initializer=initializer,
                                    name=name)


    def __call__(self, x):
        """Perform the forward pass"""
        return O.gather(self.embeddings(), x)
