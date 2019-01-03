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
    r"""Variable(s) drawn from variational distribution(s).

    TODO: more... This is an :math:`\alpha=3` inline alpha

    And here is a:

    .. math::

        y \sim \mathcal{N}(0, 1)

    non-inline math

    Parameters
    ----------
    shape : int, list of ints, or 1D np.ndarray, optional
        Shape of the array containing the variables. 
        Default = 1
    name : str, optional
        Name of the variable(s).  
        Default = 'Variable'
    prior_fn : None or `bk.distribution`
        Prior probability distribution function.  
        Default = Normal
    prior_params : np.ndarray, list of ints, floats, or `np.ndarray`s, optional
        Parameters of the prior distribution.  To use different prior parameter
        values for each element of the variable array, pass a list of 
        np.ndarrays, where each has shape matching the `shape` arg.
        Default = [0, 1] (assumes `prior_fn` = `Normal`)
    posterior_fn : `bk.distribution`, optional
        Probability distribution function to use to approximate the posterior.
        Must be a distribution from the location-scale family (such as Normal, 
        StudentT, Cauchy).
        Default = Normal
    post_param_names : list of strings, optional
        List of posterior distribution parameter names.  Elements in this 
        list should correspond to elements of `post_param_lb` and 
        `post_param_ub`.
        Default = ['loc', 'scale'] (assumes `posterior_fn` = `Normal`)
    post_param_lb : list of ints or floats or `None`s, optional
        List of posterior distribution parameter lower bounds.  The 
        variational distribution's i-th unconstrained parameter value will 
        be transformed to fall between `post_param_lb[i]` and 
        `post_param_ub[i]`. Elements of this list should correspond to 
        elements of `post_param_names` and `post_param_ub`.
        Default = [None, 0] (assumes `posterior_fn` = `Normal`)
    post_param_ub : list of ints or floats or `None`s, optional
        List of posterior distribution parameter upper bounds.  The 
        variational distribution's i-th unconstrained parameter value will 
        be transformed to fall between `post_param_lb[i]` and 
        `post_param_ub[i]`. Elements of this list should correspond to 
        elements of `post_param_names` and `post_param_ub`.
        Default = [None, None] (assumes `posterior_fn` = `Normal`)
    lb : int, float, or None, optional
        Lower bound for the variable(s).  The unconstrained posterior 
        distribution(s) will be transformed to lie between `lb` and `ub`.
        Default = None
    ub : int, float, or None, optional
        Upper bound for the variable(s).  The unconstrained posterior 
        distribution(s) will be transformed to lie between `lb` and `ub`.
        Default = None
    seed : int, float, or None, optional
        Seed for the random number generator (a `tfd.SeedStream`).  Set to 
        `None` to use the global seed.
        Default = None
    estimator : {'flipout' or None}, optional
        Method of posterior estimator to use. Valid methods:

        * None: Generate random samples from the variational distribution 
          for each batch independently.
        * 'flipout': Use the Flipout estimator [1]_ to more efficiently 
          generate samples from the variational distribution.

        Default = 'flipout'

    Examples
    --------
    TODO

    Notes
    -----
    When using the flipout estimator (`estimator='flipout'`), `posterior_fn` 
    must be a symmetric distribution of the location-scale family - one of:

    * `bk.Normal`
    * `bk.StudentT`
    * `bk.Cauchy`

    References
    ----------
    .. [1] Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. 
         Flipout: Efficient Pseudo-Independent Weight Perturbations on 
         Mini-Batches. _International Conference on Learning Representations_, 
         2018. https://arxiv.org/abs/1803.04386
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
        """Construct an array of variable(s)."""

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
        assert prior_fn is None or isinstance(prior_fn, BaseDistribution), \
            'prior_fn must be a bk distribution or None'
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
                                   np.zeros(shape).shape[::-1])), \
                        'prior_params must be broadcastable to shape'
        assert isinstance(posterior_fn, BaseDistribution), \
            'posterior_fn must be a bk distribution'
        assert isinstance(post_param_names, list), \
            'post_param_names must be a list of strings'        
        assert all(isinstance(n, str) for n in post_param_names), \
            'post_param_names must be a list of strings'
        assert isinstance(post_param_lb, list), \
            'post_param_lb must be a list of numbers'
        assert len(post_param_lb)==len(post_param_names),\
            'post_param_lb must be same length as post_param_names'
        for p_lb in post_param_lb:
            assert p_lb is None or isinstance(p_lb, [int, float]), \
                'post_param_lb must contain ints or floats or None'
        assert isinstance(post_param_ub, list), \
            'post_param_ub must be a list of numbers'
        assert len(post_param_ub)==len(post_param_names),\
            'post_param_ub must be same length as post_param_names'
        for p_ub in post_param_ub:
            assert p_ub is None or isinstance(p_ub, [int, float]), \
                'post_param_ub must contain ints or floats or None'
        assert lb is None or isinstance(lb, [int, float]), \
            'lb must be None, int, or float'
        assert ub is None or isinstance(ub, [int, float]), \
            'ub must be None, int, or float'
        assert estimator is None or isinstance(estimator, str), \
            'estimator must be None or a string'

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
        self.estimator = estimator
        self.posterior = None


    def _bound(self, data, lb, ub):
        """Bound data by applying a transformation.

        TODO: docs... explain exp for bound on one side, sigmoid for both lb+ub

        Parameters
        ----------
        data : `tf.Tensor`
            Data to bound between `lb` and `ub`.
        lb : None, int, float, or `tf.Tensor` broadcastable with `data`
            Lower bound.
        ub : None, int, float, or `tf.Tensor` broadcastable with `data`
            Upper bound.

        Returns
        -------
        bounded_data : `tf.Tensor`
            The data after being transformed.
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


    def _ensure_is_built(self):
        """Raises a RuntimeError if variable has not yet been built."""
        if self.posterior is None:
            raise RuntimeError('variable must first be built')
    

    def build(self, data):
        """Build the layer.

        TODO: docs

        Parameters
        ----------
        data : `tf.Tensor`
            Data for this batch.
        """

        # Build the prior distribution
        if self.prior_fn is not None:
            prior = self.prior_fn(*self.prior_params)
            prior.build(data)
            self.prior = prior.built_obj
        else:
            self.prior = None

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
        
        TODO: docs

        Parameters
        ----------
        data : `tf.Tensor`
            Data for this batch.
        """

        # Ensure variable has been built
        self._ensure_is_built()

        # Compute batch shape
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
            if not isinstance(self.posterior, [tfd.Normal, 
                                               tfd.StudentT,
                                               tfd.Cauchy]):
                raise ValueError('flipout requires a symmetric posterior ' +
                                 'distribution in the location-scale family')

            # Posterior mean
            w_mean = self._bound(self.posterior.mean(), lb, ub)

            # Sample from centered posterior distribution
            w_sample = self.posterior.sample(seed=seed_stream()) - w_mean

            # Random sign matrixes
            sign_r = random_rademacher(w_sample.shape, dtype=data.dtype,
                                       seed=seed_stream())
            sign_s = random_rademacher(batch_shape, dtype=data.dtype,
                                       seed=seed_stream())

            # Flipout-generated samples for this batch
            samples = tf.multiply(tf.expand_dims(w_sample*sign_r, 0), sign_s)
            samples += tf.expand_dims(w_mean, 0)

        # No other estimators supported at the moment
        else:
            raise ValueError('estimator must be None or flipout')

        # Apply bounds and return
        return self._bound(samples, lb, ub)


    def mean(self, data):
        """Mean of the variational distribution.

        TODO: docs

        Parameters
        ----------
        data : `tf.Tensor`
            Data for this batch.
        """
        self._ensure_is_built()
        return self._bound(self.posterior.mean(), lb, ub)


    def log_loss(self, vals):
        """Loss due to prior.

        TODO: docs

        Parameters
        ----------
        vals : `tf.Tensor`
            Values which were sampled from the variational distribution.
        """
        self._ensure_is_built()
        if self.prior is not None:
            return tf.reduce_sum(self.prior.log_prob(vals))
            # TODO: have to add KL term?
        else:
            return 0 #no prior, no loss


# TODO: above assumes a continuous variable, should add support for
# discrete variables.  In theory can just set posterior_fn to 
# Bernoulli or Categorical, and make mean() return the mode?