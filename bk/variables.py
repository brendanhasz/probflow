"""Variables.

TODO: more info...

"""



from abc import ABC, abstractmethod
from scipy import stats
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from core import BaseLayer
from distributions import BaseDistribution, Normal



# ensure prior is a BaseDistribution


class Variable():
    """TODO Variational variable


    TODO: More info...

    Additional kwargs include lb, ub, 

    """

    def __init__(self, 
                 shape=[1],
                 prior_fn=Normal,
                 prior_args=[0, 1],
                 lb=None,
                 ub=None):
        """Construct variable.

        TODO: docs.

        """

        # Check types
        assert isinstance(prior_fn, BaseDistribution)
        # TODO: shape must be list of ints
        # TODO: prior_args must be list of ???

        # Assign attributes
        self.shape = shape
        self.prior_fn = prior_fn
        self.prior_args = prior_args
        self.lb = lb
        self.ub = ub
    

    def build(self, data):
        """Build the layer."""

        # Build the prior distribution
        prior = self.prior_fn(*self.prior_args)
        prior.build(data)
        self.prior = prior.built_obj

        # Compute size of batch
        self.batch_size = data.shape[0]

        # Create mean and std parameters of variational distribution
        loc = tf.get_variable('loc', shape=self.shape)
        scale = tf.get_variable('scale', shape=self.shape)

        # Transform scale parameter to ensure >0
        scale = tf.exp(scale)

        # Create variational posterior distribution
        self.posterior = tfd.Normal(loc=loc, scale=scale)

        # TODO: handle when lb (lower bound) or ub (upper bound) is not empty

        # TODO: handle arbitrary types of posterior distributions?


    def sample(self):
        """Sample from the variational distribution."""
        # TODO: should return a tensor generated w/ flipout w/ correct batch shape


    def mean(self):
        """Mean of the variational distribution."""
        return self.posterior.mean()


    def log_loss(self, vals):
        """Loss due to prior."""
        return tf.reduce_sum(self.prior.log_prob(vals))
