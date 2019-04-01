"""Parameters.

TODO: more info...

----------

"""

__all__ = [
    'Parameter',
    'ScaleParameter',
]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher

from .core import BaseParameter, BaseDistribution
from .distributions import Normal, StudentT, Cauchy, InvGamma
from .utils.plotting import plot_dist, centered_text


class Parameter(BaseParameter):
    r"""Parameter(s) drawn from variational distribution(s).

    TODO: describe...


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
    posterior : |Distribution|
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    seed : int, float, or |None|
        Seed for the random number generator.
        Set to |None| to use the global seed.
        Default = |None|
    transform : lambda function
        Transform to apply to the random variable.  For example, to create a
        parameter with an inverse gamma posterior, use
        ``posterior``=:class:`.Gamma`` and
        ``transform = lambda x: tf.reciprocal(x)``
        Default is to use no transform.
    inv_transform : lambda function
        Inverse transform which will convert values in transformed space back
        into the posterior distribution's coordinates.  For example, to create
        a parameter with an inverse gamma posterior, use
        ``posterior``=:class:`.Gamma``,
        ``transform = lambda x: tf.reciprocal(x)``, and
        ``inv_transform = lambda x: tf.reciprocal(x)``.
        Default is to use no transform.
    initializer : {|None| or dict or |Tensor| or |Initializer|}
        Initializer for each variational posterior parameter.  To use the same
        initializer for each variational posterior parameter, pass a |Tensor|
        or an |Initializer|.  Set a different initializer for each variational
        posterior parameter by passing a dict with keys containing the 
        parameter names, and values containing the |Tensor| or |Initializer| 
        with which to initialize each parameter.
        Default is to use the default initializer for that |Distribution|.


    Examples
    --------
    TODO


    """

    def __init__(self,
                 shape=1,
                 name='Parameter',
                 prior=Normal(0, 1),
                 posterior=Normal,
                 seed=None,
                 transform=lambda x: x,
                 inv_transform=lambda x: x,
                 initializer=None):
        """Construct an array of Parameter(s)."""

        # Check types
        if not isinstance(shape, (int, list, np.ndarray)):
            raise TypeError('shape must be int, list of ints, or ndarray')
        if isinstance(shape, int) and shape < 1:
            raise ValueError('shape must be positive')
        if isinstance(shape, list):
            for t_shape in shape:
                if not isinstance(t_shape, int):
                    raise TypeError('each element of shape must be an int')
        if isinstance(shape, np.ndarray):
            if shape.dtype.char not in np.typecodes['AllInteger']:
                raise TypeError('shape must be int(s)')
            if not np.all(shape >= 0):
                raise ValueError('shape must be positive')
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        if prior is not None and not isinstance(prior, BaseDistribution):
            raise TypeError('prior must be None or a probflow distribution')
        if not issubclass(posterior, BaseDistribution):
            raise TypeError('posterior must be a probflow distribution')
        init_types = (dict, tf.Tensor, tf.keras.initializers.Initializer)
        if initializer is not None and not isinstance(initializer, init_types):
            raise TypeError('initializer must be None, a Tensor, an'
                            ' Initializer, or a dict')
        if isinstance(initializer, dict):
            init_types = (float, np.ndarray, tf.Tensor, 
                          tf.keras.initializers.Initializer)
            for arg in initializer:
                if (initializer[arg] is not None and
                    not isinstance(initializer[arg], init_types)):
                    raise TypeError('each value in initializer dict must be '
                                    'None, a Tensor, or an Initializer')

        # Make shape a list
        if isinstance(shape, int):
            shape = [shape]
        if isinstance(shape, np.ndarray):
            shape = shape.tolist()

        # Assign attributes
        self.shape = shape
        self.name = name
        self.prior = prior
        self.posterior_fn = posterior
        self.seed = seed
        self.transform = transform
        self.inv_transform = inv_transform
        self._built_posterior = None
        self._session = None
        self._is_built = False
        self.initializer = initializer


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
                return ub - tf.exp(-data)
            else:
                return lb + (ub-lb)*tf.sigmoid(data) # [lb, ub]


    def _build_recursively(self, data, batch_shape):
        """Build the parameter.

        TODO: docs

        Parameters
        ----------
        data : |Tensor|
            Data for this batch.
        """
        self._make_name_unique()
        self._build_prior(data, batch_shape)
        self._build_posterior(data, batch_shape)
        self._build_mean()
        self._build_sample(data, batch_shape)
        self._build_losses()
        self._is_built = True


    def _make_name_unique(self):
        """Ensure this parameter's name is a unique scope name in TF graph."""
        # TODO: getting an error here if you try to make duplicate *non-default* names
        new_name = self.name
        ix = 1
        while tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=new_name):
            new_name = self.name + '_' + str(ix)
            ix += 1
        self.name = new_name


    def _build_prior(self, data, batch_shape):
        """Build the parameter's prior distribution."""
        if self.prior is not None:
            self.prior._build_recursively(data, batch_shape)
            self._built_prior = self.prior.built_obj
            # TODO: Check that the built prior shape is broadcastable w/ self.shape


    def _build_posterior(self, data, batch_shape):
        """Build the parameter's posterior distribution."""

        # Convert float initializer values to matching type
        if isinstance(self.initializer, dict):
            for param in self.initializer:
                if isinstance(self.initializer[param], (float, np.ndarray)):
                    self.initializer[param] = \
                        tf.constant(self.initializer[param], dtype=data.dtype)
        elif isinstance(self.initializer, (float, np.ndarray)):
            self.initializer = tf.constant(self.initializer, dtype=data.dtype)

        # Create posterior distribution parameters
        params = dict()
        with tf.variable_scope(self.name):
            for arg in self.posterior_fn._post_param_bounds:
                if self.initializer is None: #use default initializer
                    init = self.posterior_fn._post_param_init[arg]
                    params[arg] = tf.get_variable(arg, shape=self.shape,
                                                  initializer=init)
                elif isinstance(self.initializer, dict):
                    params[arg] = \
                        tf.get_variable(arg, initializer=self.initializer[arg])
                else:
                    params[arg] = \
                        tf.get_variable(arg, initializer=self.initializer)

        # Transform posterior parameters
        for arg in self.posterior_fn._post_param_bounds:
            lb = self.posterior_fn._post_param_bounds[arg][0]
            ub = self.posterior_fn._post_param_bounds[arg][1]
            params[arg] = self._bound(params[arg], lb, ub)

        # Create variational posterior distribution
        self._params = params
        self.posterior = self.posterior_fn(**params)
        self.posterior._build_recursively(data, batch_shape)
        self._built_posterior = self.posterior.built_obj


    def _build_mean(self):
        """Build the mean model."""
        self._mean_obj_raw = tf.expand_dims(self._built_posterior.mean(), 0)
        self.mean_obj = self.transform(self._mean_obj_raw)


    def _build_sample(self, data, batch_shape):
        """Build the sample model."""

        # Seed generator
        seed_stream = tfd.SeedStream(self.seed, salt=self.name)

        # Draw random samples from the posterior
        samples = self._built_posterior.sample(sample_shape=batch_shape,
                                               seed=seed_stream())

        # Apply transformation
        self._built_obj_raw = samples
        self.built_obj = self.transform(self._built_obj_raw)


    def _build_losses(self):
        """Build all the losses."""
        if self.prior is None: #no prior, no losses
            self._log_loss = 0
            self._mean_log_loss = 0
            self._kl_loss = 0
        else:
            reduce_dims = np.arange(1, self._built_obj_raw.shape.ndims)
            self._log_loss = tf.reduce_sum(
                self._built_prior.log_prob(self._built_obj_raw) +
                self.prior.samp_loss_sum,
                axis=reduce_dims)
            self._mean_log_loss = tf.reduce_sum(
                self._built_prior.log_prob(self._mean_obj_raw) +
                self.prior.mean_loss_sum)
            self._kl_loss = tf.reduce_sum(
                tfd.kl_divergence(self._built_posterior,
                                  self._built_prior) +
                self.prior.kl_loss_sum)


    def _ensure_is_built(self):
        """Raises a RuntimeError if parameter has not yet been built."""
        if not self._is_built:
            raise RuntimeError('parameter must first be built')


    def _ensure_is_fit(self):
        """Raises a RuntimeError if parameter's modelhas not yet been fit."""
        if self._session is None:
            raise RuntimeError('model must first be fit')


    def posterior_mean(self):
        """Get the mean of the posterior distribution(s).

        TODO: docs: returns a numpy array

        Returns
        -------
        |ndarray|
            Mean of the parameter posterior distribution.  Size ``self.shape``.
        """
        self._ensure_is_built()
        self._ensure_is_fit()
        mean_op = self._built_posterior.mean()
        mean_op = self.transform(mean_op)
        mean = self._session.run(mean_op)
        return mean


    def sample_posterior(self, num_samples=1000):
        """Sample from the posterior distribution.

        TODO: this is similar to _sample(), but returns a numpy array
        (meant to be used by the user to examine the posterior dist)

        Returns
        -------
        |ndarray|
            Samples from the parameter posterior distribution.  Of size
            ``(num_samples,self.shape)``.

        """
        self._ensure_is_built()
        self._ensure_is_fit()
        samples_op = self._built_posterior.sample(sample_shape=num_samples)
        samples_op = self.transform(samples_op)
        samples = self._session.run(samples_op)
        return samples


    def sample_prior(self, num_samples=1000):
        """Sample from the prior distribution.

        TODO: docs

        Returns
        -------
        |ndarray|
            Samples from the parameter prior distribution.  Of size
            ``(num_samples,self.shape)``.  If this parameter has not prior, 
            returns an empty list.
        """

        # Return empty list if there is no prior
        if self.prior is None:
            return []

        # Sample from the prior distribution
        self._ensure_is_built()
        self._ensure_is_fit()
        samples_op = self._built_prior.sample(sample_shape=num_samples)
        samples_op = self.transform(samples_op)
        samples = self._session.run(samples_op)
        return samples


    def plot_posterior(self, num_samples=1000, style='fill', bins=20, ci=0.0,
                       bw=0.075, alpha=0.4, color=None):
        """Plot distribution of samples from the posterior distribution.

        TODO: docs

        Parameters
        ----------
        num_samples : int
            Number of samples to take from each posterior distribution for
            estimating the density.  Default = 1000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        bins : int or list or |ndarray|
            Number of bins to use for the posterior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram
        """

        # Check inputs
        if type(num_samples) is not int or num_samples < 1:
            raise TypeError('num_samples must be an int greater than 0')
        if type(style) is not str or style not in ['fill', 'line', 'hist']:
            raise TypeError("style must be \'fill\', \'line\', or \'hist\'")
        if not isinstance(bins, (int, float, np.ndarray)):
            raise TypeError('bins must be an int or list or numpy vector')
        if type(ci) is not float or ci<0.0 or ci>1.0:
            raise TypeError('ci must be a float between 0 and 1')
        if type(alpha) is not float or alpha<0.0 or alpha>1.0:
            raise TypeError('alpha must be a float between 0 and 1')

        # Sample from the posterior
        samples = self.sample_posterior(num_samples=num_samples)
        
        # Plot the posterior densities
        plot_dist(samples, xlabel=self.name, style=style, bins=bins, 
                  ci=ci, bw=bw, alpha=alpha, color=color)


    def plot_prior(self, num_samples=10000, style='fill', bins=20, ci=0.0,
                       bw=0.075, alpha=0.4, color=None):
        """Plot distribution of samples from the prior distribution.

        TODO: docs

        NOTE that really you could have just evaluated the prior fn @ x values
        and plotted a deterministic line.
        BUT that only works for simple priors (e.g. Normal(0,1))
        Since probflow allows (non-deterministic) parameterized priors, e.g.:
        prior = Parameter()*Input()+Parameter()
        it's simpler just to sample here instead of checking which is the case

        Parameters
        ----------
        num_samples : int
            Number of samples to take from each prior distribution for
            estimating the density.  Default = 1000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        bins : int or list or |ndarray|
            Number of bins to use for the prior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        """

        # Show "No prior"
        if self.prior is None:
            centered_text('No prior on '+self.name)
            return

        # Sample from the posterior
        samples = self.sample_prior(num_samples=num_samples)
        
        # Plot the posterior densities
        plot_dist(samples, xlabel='Prior on '+self.name, style=style, 
                  bins=bins, ci=ci, bw=bw, alpha=alpha, color=color)


    def __str__(self, prepend=''):
        """String representation of a parameter."""
        # TODO: will have to change this to allow complicated priors
        return (prepend + 'Parameter \'' + self.name+'\'' +
                ' shape=' + str(tuple(self.shape)) + 
                ' prior=' + str(self.prior).replace(' ', '') +
                ' posterior=' + self.posterior_fn.__name__)


    def __getitem__(self, inds):
        """Get parameters by index."""
        from .layers import Gather
        return Gather(self, inds)


    def __lshift__(self, dist):
        """Set the prior distribution for this parameter."""

        # Ensure prior to set is a distribution
        if dist is not None and not isinstance(dist, BaseDistribution):
            raise TypeError('prior must be a distribution object or None')

        # Set new prior
        self.prior = dist



class ScaleParameter(Parameter):
    r"""Standard deviation parameter.

    This is a convenience class for creating a standard deviation parameter
    (:math:`\sigma`).  It is created by first constructing a variance 
    parameter (:math:`\sigma^2`) which uses an inverse gamma distribution as
    the variational posterior.

    .. math::

        \sigma^2 \sim \text{InvGamma}(\alpha, \beta)

    Then the variance is transformed into the standard deviation:

    .. math::

        \sigma = \sqrt{\sigma^2}

    By default, a uniform prior is used.

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
        Default = |None|
    posterior : |Distribution|
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.InvGamma`
    seed : int, float, or |None|
        Seed for the random number generator.
        Set to |None| to use the global seed.
        Default = |None|
    initializer : {|None| or dict or |Tensor| or |Initializer|}
        Initializer for each variational posterior parameter.  To use the same
        initializer for each variational posterior parameter, pass a |Tensor|
        or an |Initializer|.  Set a different initializer for each variational
        posterior parameter by passing a dict with keys containing the 
        parameter names, and values containing the |Tensor| or |Initializer| 
        with which to initialize each parameter.
        Default is to initialize both the ``shape`` and ``rate`` parameters
        of the :class:`.InvGamma` variational posterior to the default for
        that distribution (see :class:`.InvGamma`).

    Examples
    --------
    TODO

    """

    def __init__(self,
                 shape=1,
                 name='ScaleParameter',
                 prior=None,
                 posterior=InvGamma,
                 seed=None,
                 initializer=None):
        super().__init__(shape=shape,
                         name=name,
                         prior=prior,
                         posterior=posterior,
                         seed=seed,
                         transform=lambda x: tf.sqrt(x),
                         inv_transform=lambda x: tf.square(x),
                         initializer=initializer)



# TODO: add support for discrete Parameters?
# In theory can just set posterior to
# Bernoulli or Categorical, and make mean() return the mode?
# and have n_categories-1 different underlying tf variables
# and transform them according to the additive logistic transformation?
# to get probs of categories
# https://en.wikipedia.org/wiki/Logit-normal_distribution#Probability_density_function_2
