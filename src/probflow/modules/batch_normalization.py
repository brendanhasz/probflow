from typing import Callable, Dict, List, Type, Union

import probflow.utils.ops as O
from probflow.distributions import Deterministic, Normal
from probflow.parameters import Parameter
from probflow.utils.base import BaseDistribution
from probflow.utils.initializers import xavier

from .module import Module


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

    def __init__(
        self,
        shape: Union[int, List[int]],
        weight_posterior: Type[BaseDistribution] = Deterministic,
        bias_posterior: Type[BaseDistribution] = Deterministic,
        weight_prior: BaseDistribution = Normal(0, 1),
        bias_prior: BaseDistribution = Normal(0, 1),
        weight_initializer: Dict[str, Callable] = {"loc": xavier},
        bias_initializer: Dict[str, Callable] = {"loc": xavier},
        name="BatchNormalization",
    ):

        # Create the parameters
        self.weight = Parameter(
            shape=shape,
            posterior=weight_posterior,
            prior=weight_prior,
            initializer=weight_initializer,
            name=name + "_weight",
        )
        self.bias = Parameter(
            shape=shape,
            posterior=bias_posterior,
            prior=bias_prior,
            initializer=bias_initializer,
            name=name + "_bias",
        )

    def __call__(self, x):
        """Perform the forward pass"""
        mean = O.mean(x, axis=0)
        std = O.std(x, axis=0)
        return self.weight() * (x - mean) / std + self.bias()
