"""
Modules are objects which take Tensor(s) as input, perform some computation on
that Tensor, and output a Tensor.  Modules can create and contain |Parameters|.
For example, neural network layers are good examples of a |Module|, since 
they store parameters, and use those parameters to perform a computation
(the forward pass of the data through the layer).

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



from typing import Union, List, Dict, Callable, Type

import probflow.core.ops as O
from probflow.core.settings import get_flipout
from probflow.core.settings import get_backend
from probflow.core.base import *
from probflow.distributions import Deterministic
from probflow.distributions import Normal
from probflow.parameters import Parameter
from probflow.utils.initializers import xavier
from probflow.utils.initializers import scale_xavier



class Module(BaseModule):
    r"""Abstract base class for Modules.

    TODO

    """

    def _params(self, obj):
        """Recursively search for |Parameters| contained within an object"""
        if isinstance(obj, BaseParameter):
            return [obj]
        elif isinstance(obj, BaseModule):
            return obj.parameters
        elif isinstance(obj, list):
            return self._list_params(obj)
        elif isinstance(obj, dict):
            return self._dict_params(obj)
        else:
            return []


    def _list_params(self, the_list: List):
        """Recursively search for |Parameters| contained in a list"""
        return [p for e in the_list for p in self._params(e)]


    def _dict_params(self, the_dict: Dict):
        """Recursively search for |Parameters| contained in a dict"""
        return [p for _, e in the_dict.items() for p in self._params(e)]


    @property
    def parameters(self):
        """A list of |Parameters| in this |Module| and its sub-Modules."""
        return [p for _, a in vars(self).items() for p in self._params(a)]


    @property
    def modules(self):
        """A list of sub-Modules in this |Module|, including itself."""
        return [m for a in vars(self).values()
                if isinstance(a, BaseModule)
                for m in a.modules] + [self]


    @property
    def trainable_variables(self):
        """A list of trainable backend variables within this |Module|"""
        return [v for p in self.parameters for v in p.trainable_variables]
        # TODO: look for variables NOT in parameters too
        # so users can mix-n-match tf.Variables and pf.Parameters in modules 


    @property
    def n_parameters(self):
        """Get the number of independent parameters of this module"""
        return sum([p.n_parameters for p in self.parameters])


    @property
    def n_variables(self):
        """Get the number of underlying variables in this module"""
        return sum([p.n_variables for p in self.parameters])


    def kl_loss(self):
        """Compute the sum of the Kullback-Leibler divergences between
        priors and their variational posteriors for all |Parameters| in this
        |Module| and its sub-Modules."""
        return sum([p.kl_loss() for p in self.parameters])


    def kl_loss_batch(self):
        """Compute the sum of additional Kullback-Leibler divergences due to
        data in this batch"""
        return sum([e for m in self.modules for e in m._kl_losses])


    def reset_kl_loss(self):
        """Reset additional loss due to KL divergences"""
        for m in self.modules:
            m._kl_losses = []


    def add_kl_loss(self, loss, d2=None):
        """Add additional loss due to KL divergences."""
        if d2 is None:
            self._kl_losses += [O.sum(loss, axis=None)]
        else:
            self._kl_losses += [O.sum(O.kl_divergence(loss, d2), axis=None)]



class Dense(Module):
    """Dense neural network layer.

    TODO

    Parameters
    ----------
    d_in : int
        Number of input dimensions.
    d_out : int
        Number of output dimensions (number of "units").
    name : str
        Name of this layer
    """


    def __init__(self, d_in: int, d_out: int = 1, name: str = 'Dense'):

        # Check types
        if d_in < 1:
            raise ValueError('d_in must be >0')
        if d_out < 1:
            raise ValueError('d_out must be >0')

        # Create the parameters
        self.d_in = d_in
        self.d_out = d_out
        self.weights = Parameter(shape=[d_in, d_out], name=name+'_weights')
        self.bias = Parameter(shape=[1, d_out], name=name+'_bias')


    def __call__(self, x):
        """Perform the forward pass"""

        # Using the Flipout estimator
        if get_flipout():
        
            # With PyTorch
            if get_backend() == 'pytorch':
                raise NotImplementedError

            # With Tensorflow
            else:

                import tensorflow as tf
                import tensorflow_probability as tfp

                # Flipout-estimated weight samples
                s = tfp.python.math.random_rademacher(tf.shape(x))
                r = tfp.python.math.random_rademacher([x.shape[0],self.d_out])
                norm_samples = tf.random.normal([self.d_in, self.d_out])
                w_samples = self.weights.variables['scale'] * norm_samples
                w_noise = r*((x*s) @ w_samples)
                w_outputs = x @ self.weights.variables['loc'] + w_noise
                
                # Flipout-estimated bias samples
                r = tfp.python.math.random_rademacher([x.shape[0],self.d_out])
                norm_samples = tf.random.normal([self.d_out])
                b_samples = self.bias.variables['scale'] * norm_samples
                b_outputs = self.bias.variables['loc'] + r*b_samples
                
                return w_outputs + b_outputs
        
        # Without Flipout
        else:
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


    def __init__(self, steps: List[Callable], name: str = 'Sequential'):

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
                 shape: Union[int, List[int]],
                 weight_posterior: Type[BaseDistribution] = Deterministic,
                 bias_posterior: Type[BaseDistribution] = Deterministic,
                 weight_prior: BaseDistribution = Normal(0, 1),
                 bias_prior: BaseDistribution = Normal(0, 1),
                 weight_initializer: Dict[str, Callable] = {'loc': xavier},
                 bias_initializer: Dict[str, Callable] = {'loc': xavier},
                 name='BatchNormalization'):

        # Create the parameters
        self.weight = Parameter(shape=shape,
                                posterior=weight_posterior,
                                prior=weight_prior,
                                initializer=weight_initializer,
                                name=name+'_weight')
        self.bias = Parameter(shape=shape,
                              posterior=bias_posterior,
                              prior=bias_prior,
                              initializer=bias_initializer,
                              name=name+'_bias')


    def __call__(self, x):
        """Perform the forward pass"""
        mean = O.mean(x, axis=0)
        std = O.std(x, axis=0)
        return self.weight()*(x-mean)/std+self.bias()



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
    k : int > 0 or List[int]
        Number of categories to embed.
    d : int > 0 or List[int]
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
                 k: Union[int, List[int]],
                 d: Union[int, List[int]],
                 posterior: Type[BaseDistribution] = Deterministic,
                 prior: BaseDistribution = Normal(0, 1),
                 initializer: Dict[str, Callable] = {'loc': xavier},
                 name: str = 'Embeddings'):

        # Convert to list if not already
        if isinstance(k, int):
            k = [k]
        if isinstance(d, int):
            d = [d]

        # Check values
        if len(k) != len(d):
            raise ValueError('d and k must be the same length')
        if any(e<1 for e in k):
            raise ValueError('k must be >0')
        if any(e<1 for e in d):
            raise ValueError('d must be >0')

        # Create the parameters
        self.embeddings = [Parameter(shape=[k[i], d[i]],
                                     posterior=posterior,
                                     prior=prior,
                                     initializer=initializer,
                                     name=name+'_'+str(i))
                           for i in range(len(d))]


    def __call__(self, x):
        """Perform the forward pass"""
        embs = [O.gather(self.embeddings[i](), x[:, i])
                for i in range(len(self.embeddings))]
        return O.cat(embs, -1)
