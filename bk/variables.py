"""Variables.

TODO: more info...

"""



import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher

from .distributions import BaseDistribution, Normal
from .core import BaseVariable



class Variable(BaseVariable):
    """TODO Variational variable

    TODO: More info...

    Parameters
    ----------
    shape : int, list of ints, or np.ndarray
        Shape of the array containing the variables.
    name : str
        Name of the variable(s).
    prior_fn : bk distribution
        Prior probability distribution function.
    prior_params : np.ndarray, list of ints, floats, or np.ndarray s
        Parameters of the prior distribution.  You can use different prior
        parameter values for each element of the variable array by passing a 
        list of np.ndarrays, where each has shape matching the `shape` arg.

    Additional kwargs include lb, ub, 
    post_param_names - names of the posterior distribution parameters
    post_param_lb - lower bounds for posterior dist params
    post_param_ub - upper bounds for posterior dist params
    lb - lower bound for this variable
    ub - upper bound for this variable
    """

    def __init__(self, 
                 shape=1,
                 name='Variable',
                 prior_fn=Normal,
                 prior_params=[0, 1],
                 posterior_fn=Normal,
                 post_param_names=['loc', 'scale'],
                 post_param_lb=[None, 0],
                 post_param_ub=[None, None],
                 lb=None,
                 ub=None,
                 seed=None,
                 estimator='flipout'):
        """Construct variable.

        TODO: docs.

        """

        # Check types
        assert isinstance(shape, [int, list, np.ndarray]), \
            ('shape must be an int, list of ints, or a numpy ndarray')
        if isinstance(shape, int):
            assert shape>0, 'shape must be positive'
        if isinstance(shape, list):
            for t_shape in shape:
                assert isinstance(t_shape, int), 'shape must be integer(s)'
        if isinstance(shape, np.ndarray):
            assert shape.dtype.char in np.typecodes['AllInteger'], \
                'shape must be integer(s)'
            assert np.all(shape>=0), 'shape must be positive'
        assert isinstance(name, str), 'name must be a string'
        assert isinstance(prior_fn, BaseDistribution), \
            'prior_fn must be a bk distribution'
        assert isinstance(prior_params, [int, float, list, np.ndarray]), \
            ('prior_params must be a np.ndarray, a list of ints, a list of' + 
             'floats, or a list of np.ndarray s')
        if isinstance(prior_params, list):
            for t_param in prior_params:
                assert isinstance(t_param, [int, float, np.ndarray]), \
                    ('prior_params must be a list of ints, floats, or' + 
                     'np.ndarray s')
                if isinstance(t_param, np.ndarray):
                    assert all((m==n) or (m==1) or (n==1) for m, n in
                               zip(t_param.shape[::-1], 
                                   np.zeros(shape).shape[::-1])) \
                        'prior_params must be broadcastable to shape'
        assert isinstance(posterior_fn, BaseDistribution), \
            'posterior_fn must be a bk distribution'
        assert isinstance(post_param_names, list), \
            'post_param_names must be a list of strings'        
        assert all(isinstance(n, str) for n in post_param_names), \
            'post_param_names must be a list of strings'

        # TODO: 
        #post_param_lb=[None, 0],
        #post_param_ub=[None, None],
        #lb=None,
        #ub=None,
        #seed=None,
        #estimator='flipout'

        # post_param_lb and _ub must be list of floats, int, None, (or np array or tensor?)
        # lb must be None, float, int, (or single np value?)
        # estimator can be None (just generate random nums for all), flipout, and that's it for now

        # Assign attributes
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
        self.seed = seed


    def _bound(self, data, lb, ub):
        """Bound data by applying a transformation.

        TODO: docs... explain exp for bound on one side, sigmoid for both lb+ub

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

        # Compute shapes
        # TODO
        batch_shape = data.shape[0]

        # Seed generator
        seed_stream = tfd.SeedStream(self.seed, salt=self.name)

        # Draw random samples from the posterior
        if self.estimator is None:
            samples = self.posterior.sample(sample_shape=batch_shape,
                                            seed=seed_stream())

        # Use the Flipout estimator (https://arxiv.org/abs/1803.04386)
        elif self.estimator=='flipout':

            # Flipout only works w/ distributions symmetric around 0
            if not isinstance(self.posterior, [tfd.Normal, tfd.StudentT, tfd.Cauchy])
                raise ValueError('flipout requires a symmetric posterior ' +
                                 'distribution in the location-scale family')

            # TODO: should return a tensor generated w/ flipout w/ correct batch shape
            # https://arxiv.org/pdf/1803.04386.pdf
            # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/layers/dense_variational.py#L687
            # initial draws from posterior are shared across samples in the mini-batch 
            #   and so should be of shape (1,?,?...)

            # TODO:

            # Posterior mean
            w_mean = self._bound(self.posterior.mean(), lb, ub)

            # Sample from centered posterior distribution
            w_sample = self.posterior.sample(seed=seed_stream()) - w_mean

            # TODO: 
            sign_in = random_rademacher(
                self.shape,
                dtype=data.dtype,
                seed=seed_stream())

            sign_out = random_rademacher(
                tf.concat([batch_shape,
                           tf.expand_dims(???, axis=0)], 0),
                dtype=data.dtype,
                seed=seed_stream())


            samples = w_mean + _matmul(_matmul(w_sample, sign_out), sign_in)
            # TODO: transpose sign_in?

            perturbed_inputs = self._matmul(
                inputs * sign_input, self.kernel_posterior_affine_tensor) * sign_output

            outputs = self._matmul(inputs, self.kernel_posterior.distribution.loc)

            outputs += perturbed_inputs

        # No other estimators supported at the moment
        else:
            raise ValueError('estimator must be None or flipout')

        # Apply bounds and return
        return self._bound(samples, lb, ub)


    def mean(self, data):
        """Mean of the variational distribution.

        TODO: docs

        """
        return self._bound(self.posterior.mean(), lb, ub)


    def log_loss(self, vals):
        """Loss due to prior.

        TODO: docs

        """
        return tf.reduce_sum(self.prior.log_prob(vals))
        # TODO: have to add KL term?
