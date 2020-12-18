from typing import Callable, Dict, List, Type, Union

import probflow.utils.ops as O
from probflow.distributions import Deterministic, Normal
from probflow.parameters import DeterministicParameter, Parameter
from probflow.utils.base import BaseDistribution
from probflow.utils.initializers import xavier

from .module import Module


class Embedding(Module):
    r"""A categorical embedding layer.

    Maps an input variable containing non-negative integers to dense vectors.
    The length of the vectors (the dimensionality of the embedding) can be set
    with the ``dims`` keyword argument.  The embedding is learned over the
    course of training: if there are N unique integers in the input, and the
    embedding dimensionality is M, a matrix of NxM free parameters is created
    and optimized to minimize the loss.

    By default, a :class:`.Deterministic` distribution is used for the
    embedding variables' posterior distributions, with :class:`.Normal`
    ``(0, 1)`` priors.  This corresponds to normal non-probabilistic embedding
    with L2 regularization.

    The embeddings can be non-probabilistic (each integer corresponds to a
    single point in M-dimensional space, the default), or probabilistic (each
    integer corresponds to a M-dimensional multivariate distribution).  Set the
    `probabilistic` kwarg to True to use probabilistic embeddings.


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
    probabilistic : bool
        Whether variational posteriors for the weights and biases should be
        probabilistic.  If False (the default), will use
        :class:`.Deterministic` distributions for the variational posteriors.
        If True, will use :class:`.Normal` distributions.
    name : str
        Name for this layer.
        Default = 'Embeddings'
    kwargs
        Additional keyword arguments are passed to the :class:`.Parameter`
        constructor which creates the embedding variables.


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

    def __init__(
        self,
        k: Union[int, List[int]],
        d: Union[int, List[int]],
        probabilistic: bool = False,
        name: str = "Embedding",
        **kwargs
    ):

        # Convert to list if not already
        if isinstance(k, int):
            k = [k]
        if isinstance(d, int):
            d = [d]

        # Check values
        if len(k) != len(d):
            raise ValueError("d and k must be the same length")
        if any(e < 1 for e in k):
            raise ValueError("k must be >0")
        if any(e < 1 for e in d):
            raise ValueError("d must be >0")

        # Override posterior and initializer for probabilistic embedding
        if probabilistic:
            ParameterClass = Parameter
        else:
            ParameterClass = DeterministicParameter

        # Create the parameters
        self.embeddings = [
            ParameterClass(
                shape=[k[i], d[i]], name=name + "_" + str(i), **kwargs
            )
            for i in range(len(d))
        ]

    def __call__(self, x):
        """Perform the forward pass"""
        embs = [
            O.gather(self.embeddings[i](), x[:, i])
            for i in range(len(self.embeddings))
        ]
        return O.cat(embs, -1)
