"""Parameters.

TODO: more info...

----------

"""



import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher

from .core import BaseParameter, BaseDistribution
from .distributions import Normal, InvGamma


__all__ = [
    'Parameter',
    'ScaleParameter',
]


class Parameter(BaseParameter):
    r"""Parameter(s) drawn from variational distribution(s).

    TODO: describe...

    .. math::

        y \sim \mathcal{N}(0, 1)


    Parameters
    ----------
    shape : int, list of int, or 1D |ndarray|
        Shape of the array containing the parameters. 
        Default = ``1``
    name : str
        Name of the parameter(s).  
        Default = ``'Parameter'``
    prior : |None| or a |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0,1)``
    posterior_fn : |Distribution|
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    post_param_names : list of str
        List of posterior distribution parameter names.  Elements in this 
        list should correspond to elements of ``post_param_lb`` and 
        ``post_param_ub``.
        Default = ``['loc', 'scale']`` (assumes ``posterior_fn = Normal``)
    post_param_lb : list of {int or float or |None|}
        List of posterior distribution parameter lower bounds.  The 
        variational distribution's ``i``-th unconstrained parameter value will 
        be transformed to fall between ``post_param_lb[i]`` and 
        ``post_param_ub[i]``. Elements of this list should correspond to 
        elements of ``post_param_names`` and ``post_param_ub``.
        Default = ``[None, 0]`` (assumes ``posterior_fn = Normal``)
    post_param_ub : list of {int or float or |None|}
        List of posterior distribution parameter upper bounds.  The 
        variational distribution's ``i``-th unconstrained parameter value will 
        be transformed to fall between ``post_param_lb[i]`` and 
        ``post_param_ub[i]``. Elements of this list should correspond to 
        elements of ``post_param_names`` and ``post_param_ub``.
        Default = ``[None, None]`` (assumes ``posterior_fn = Normal``)
    seed : int, float, or |None|
        Seed for the random number generator.  
        Set to |None| to use the global seed.
        Default = |None|
    estimator : {``'flipout'`` or |None|}
        Method of posterior estimator to use. Valid values:

        * |None|: Generate random samples from the variational distribution 
          for each batch independently.
        * `'flipout'`: Use the Flipout estimator :ref:`[1] <ref_flipout>` to 
          more efficiently generate samples from the variational distribution.

        Default = ``'flipout'``

    Notes
    -----
    When using the flipout estimator (``estimator='flipout'``), ``posterior_fn`` 
    must be a symmetric distribution of the location-scale family - one of:

    * :class:`.Normal`
    * :class:`.StudentT`
    * :class:`.Cauchy`

    Examples
    --------
    TODO

    References
    ----------
    .. _ref_flipout:
    .. [1] Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse. 
        Flipout: Efficient Pseudo-Independent Weight Perturbations on 
        Mini-Batches. *International Conference on Learning Representations*, 
        2018. https://arxiv.org/abs/1803.04386
    """

    def __init__(self, 
                 shape=1,
                 name='Parameter',
                 prior=Normal(0, 1),
                 posterior_fn=Normal,
                 post_param_names=['loc', 'scale'],
                 post_param_lb=[None, 0],
                 post_param_ub=[None, None],
                 seed=None,
                 estimator='flipout'):
        """Construct an array of Parameter(s)."""

        # Check types
        assert isinstance(shape, (int, list, np.ndarray)), \
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
        assert prior is None or isinstance(prior, BaseDistribution), \
            'prior must be a probflow distribution or None'
        assert issubclass(posterior_fn, BaseDistribution), \
            'posterior_fn must be a probflow distribution'
        assert isinstance(post_param_names, list), \
            'post_param_names must be a list of strings'        
        assert all(isinstance(n, str) for n in post_param_names), \
            'post_param_names must be a list of strings'
        assert isinstance(post_param_lb, list), \
            'post_param_lb must be a list of numbers'
        assert len(post_param_lb)==len(post_param_names),\
            'post_param_lb must be same length as post_param_names'
        for p_lb in post_param_lb:
            assert p_lb is None or isinstance(p_lb, (int, float)), \
                'post_param_lb must contain ints or floats or None'
        assert isinstance(post_param_ub, list), \
            'post_param_ub must be a list of numbers'
        assert len(post_param_ub)==len(post_param_names),\
            'post_param_ub must be same length as post_param_names'
        for p_ub in post_param_ub:
            assert p_ub is None or isinstance(p_ub, (int, float)), \
                'post_param_ub must contain ints or floats or None'
        assert estimator is None or isinstance(estimator, str), \
            'estimator must be None or a string'

        # Assign attributes
        self.shape = shape
        self.name = name
        self.prior = prior
        self.posterior_fn = posterior_fn
        self.post_param_names = post_param_names
        self.post_param_lb = post_param_lb
        self.post_param_ub = post_param_ub
        self.seed = seed
        self.estimator = estimator
        self.posterior = None

        # TODO: initializer?

        # TODO: dtype?


    def _bound(self, data, lb, ub):
        """Bound data by applying a transformation.

        TODO: docs... explain exp for bound on one side, sigmoid for both lb+ub

        Parameters
        ----------
        data : |Tensor|
            Data to bound between ``lb`` and ``ub``.
        lb : |None|, int, float, or |Tensor| broadcastable with ``data``
            Lower bound.
        ub : |None|, int, float, or |Tensor| broadcastable with ``data``
            Upper bound.

        Returns
        -------
        bounded_data : |Tensor|
            The data after being transformed.
        """
        if ub is None:
            if lb is None:
                return data # [-Inf, Inf]
            else: 
                return lb + tf.exp(data) # [lb, Inf]
        else:
            if lb is None: #negative # [-Inf, ub]
                return tf.exp(-data)
            else: 
                return lb + (ub-lb)*tf.sigmoid(data) # [lb, ub]


    def _ensure_is_built(self):
        """Raises a RuntimeError if parameter has not yet been built."""
        if self.posterior is None:
            raise RuntimeError('parameter must first be built')
    

    def _build(self, data):
        """Build the layer.

        TODO: docs

        Parameters
        ----------
        data : |Tensor|
            Data for this batch.
        """

        # Build the prior distribution
        if self.prior is not None:
            self.prior.build(data)
            self._built_prior = self.prior.built_obj
            # TODO: Check that the built prior shape is broadcastable w/ self.shape

        # Create posterior distribution parameters
        params = dict()
        with tf.variable_scope(self.name):
            for arg in self.post_param_names:
                params[arg] = tf.get_variable(arg, shape=self.shape)

        # Transform posterior parameters
        for arg, lb, ub in zip(self.post_param_names, 
                               self.post_param_lb, 
                               self.post_param_ub):
            params[arg] = self._bound(params[arg], lb, ub)

        # Create variational posterior distribution
        self.posterior = self.posterior_fn(**params)
        self.posterior.build(data)
        self._built_posterior = self.posterior.built_obj


    def _sample(self, data):
        """Sample from the variational distribution.
        
        TODO: docs

        .. admonition:: Parameter must be built first!

            Before calling :meth:`.sample` on a |Parameter|, you must first 
            :meth:`.build` it, or :meth:`.fit` a model it belongs to.

        Parameters
        ----------
        data : |Tensor|
            Data for this batch.

        Returns
        -------
        |Tensor|
            An (unevaluated) tensor with samples from the variational dist.

        """

        # Ensure parameter has been built
        self._ensure_is_built()

        # Compute batch shape
        batch_shape = data.shape[0]

        # Seed generator
        seed_stream = tfd.SeedStream(self.seed, salt=self.name)

        # Draw random samples from the posterior
        if self.estimator is None:
            samples = self._built_posterior.sample(sample_shape=batch_shape,
                                                   seed=seed_stream())

        # Use the Flipout estimator (https://arxiv.org/abs/1803.04386)
        elif self.estimator=='flipout':

            # Flipout only works w/ distributions symmetric around 0
            if not isinstance(self._built_posterior, [tfd.Normal, 
                                                      tfd.StudentT,
                                                      tfd.Cauchy]):
                raise ValueError('flipout requires a symmetric posterior ' +
                                 'distribution in the location-scale family')

            # Posterior mean
            w_mean = self._built_posterior.mean()

            # Sample from centered posterior distribution
            w_sample = self._built_posterior.sample(seed=seed_stream()) - w_mean

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
        return samples


    def _mean(self, data):
        """Mean of the variational distribution.

        TODO: docs

        .. admonition:: Parameter must be built first!

            Before calling :meth:`.mean` on a |Parameter|, you must first 
            :meth:`.build` it, or :meth:`.fit` a model it belongs to.

        Parameters
        ----------
        data : |Tensor|
            Data for this batch.
        """
        self._ensure_is_built()
        return self._built_posterior.mean()


    def _log_loss(self, vals):
        """Loss due to prior.

        TODO: docs

        .. admonition:: Parameter must be built first!

            Before calling :meth:`.log_loss` on a |Parameter|, you must first 
            :meth:`.build` it, or :meth:`.fit` a model it belongs to.

        Parameters
        ----------
        vals : |Tensor|
            Values which were sampled from the variational distribution.
        """
        self._ensure_is_built()
        if self.prior is not None:
            return (self._built_prior.log_prob(vals) + 
                    self.prior.arg_loss_sum)
        else:
            return 0 #no prior, no loss


    def _kl_loss(self):
        """Loss due to divergence between posterior and prior.

        TODO: docs...

        """
        self._ensure_is_built()
        return (tf.reduce_sum(
                    tfd.kl_divergence(self._built_posterior, 
                                      self._built_prior))
                + self.prior.kl_loss_sum)
        # TODO: make sure that the broadcasting occurs correctly here
        # eg if posterior shape is [2,1], should return 
        # (kl_div(post_1,prior_1) + kl_div(Post_2,prior_2))


    def posterior(self, num_samples=1000):
        """Sample from the posterior distribution.

        TODO: this is similar to _sample(), but returns a numpy array
        (meant to be used by the user to examine the posterior dist)

        """
        # TODO: run a tf sess and return a np array
        pass
        

    def __str__(self, prepend=''):
        """String representation of a parameter."""

        return 'Parameter \''+self.name+'\''



def ScaleParameter(self, 
                   shape=1,
                   prior=None,
                   name='ScaleParameter',
                   seed=None,
                   estimator='flipout'):
    r"""Standard deviation param whose variance has an InverseGamma posterior.

    This is a convenience function for creating a standard deviation parameter
    (\sigma).  It is created by first constructing a variance parameter 
    (:math:`\sigma^2`) which uses an inverse gamma distribution as the
    variational posterior.

    .. math::

        \sigma^2 \sim \text{InvGamma}(\alpha, \beta)

    Then the variance is transformed into the standard deviation:

    .. math::

        \sigma = \sqrt{\sigma^2}

    By default, a uniform prior is used.

    """

    # Create the variance parameter
    variance = Parameter(shape=shape,
                         name=name,
                         prior=prior,
                         posterior_fn=InvGamma,
                         post_param_names=['shape', 'rate'],
                         post_param_lb=[0, 0],
                         post_param_ub=[None, None],
                         seed=seed,
                         estimator=estimator)

    # Return the transformed parameter
    from .layers import Sqrt
    return Sqrt(variance)

    # TODO: uh but if you call .posterior() on what this returns it will
    # give you the posterior of the variance not of the std dev...


# TODO: add support for discrete Parameters?
# In theory can just set posterior_fn to 
# Bernoulli or Categorical, and make mean() return the mode?
# and have n_categories-1 different underlying tf variables
# and transform them according to the additive logistic transformation?
# to get probs of categories
# https://en.wikipedia.org/wiki/Logit-normal_distribution#Probability_density_function_2