"""Variables.

TODO: more info...

"""



from abc import ABC, abstractmethod
from scipy import stats
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from .distributions import Normal
from .core import BaseVariable



class Variable(BaseVariable):
    """TODO Variational variable

    TODO: More info...

    Additional kwargs include lb, ub, 
    post_arg_names - names of the posterior distribution parameters

    """

    def __init__(self, 
                 shape=[1],
                 name='Variable',
                 prior_fn=Normal,
                 prior_params=[0, 1],
                 posterior_fn=Normal,
                 post_param_names=['loc', 'scale'],
                 post_param_lb=[None, 0],
                 post_param_ub=[None, None],
                 lb=None,
                 ub=None):
        """Construct variable.

        TODO: docs.

        """
        self.shape = shape
        self.name = name
        self.prior_fn = prior_fn
        self.prior_params = prior_params
        self.posterior_fn = posterior_fn
        self.post_param_names = post_param_names
        self.post_param_lb = post_param_lb
        self.post_param_ub = post_param_ub
        self.lb = lb
        self.ub = ub


    def _bound(self, data, lb, ub):
        """Bound data by applying a transformation.

        TODO: docs... explain exp for >0, sigmoid for both lb+ub

        """
        if ub is None:
            if lb is None:
                return data # [-Inf, Inf]
            else: 
                return lb + tf.exp(data) # [lb, Inf]
        else:
            if lb is None: 
                #negative # [-Inf, ub]
                return tf.exp(-data)
            else: 
                return lb + (ub-lb)*tf.sigmoid(data) # [lb, ub]
    

    def build(self, data):
        """Build the layer.

        TODO: docs

        """

        # Build the prior distribution
        prior = self.prior_fn(*self.prior_params)
        prior.build(data)
        self.prior = prior.built_obj

        # Create posterior parameter variables
        params = dict()
        with tf.variable_scope(self.name):
            for arg in post_param_names:
                params[arg] = tf.get_variable(arg, shape=self.shape)

        # Transform posterior parameters
        for arg, lb, ub in zip(post_param_names, post_param_lb, post_param_ub):
            params[arg] = self._bound(params[arg], lb, ub)

        # Create variational posterior distribution
        posterior = self.posterior_fn(**params)
        posterior.build(data)
        self.posterior = posterior.built_obj


    def sample(self, data):
        """Sample from the variational distribution.
        
        """
        # TODO: should return a tensor generated w/ flipout w/ correct batch shape
        # https://arxiv.org/pdf/1803.04386.pdf
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/layers/dense_variational.py#L687

        #TODO: do we have to re-compute batch size here so it'll work w/ both validation and training?
        # Compute size of batch
        self.batch_size = data.shape[0]

        # TODO: transform generated values w/ exp (if one of ub or lb is set),
        # or with logit (if both are set)
        # so return self._bound(generated_output, self.lb, self.ub)


    def mean(self):
        """Mean of the variational distribution.

        TODO: docs

        """
        return self.posterior.mean()
        # TODO: transform w/ exp or logit if needed (depending on if lb and ub are set)
        # so return self._bound(generated_output, self.lb, self.ub)


    def log_loss(self, vals):
        """Loss due to prior.

        TODO: docs

        """
        return tf.reduce_sum(self.prior.log_prob(vals))
