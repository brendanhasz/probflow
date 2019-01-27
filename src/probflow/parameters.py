"""Parameters.

TODO: more info...

----------

"""

__all__ = [
    'Parameter',
    'ScaleParameter',
]

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher

from .core import BaseParameter, BaseDistribution
from .distributions import Normal, StudentT, Cauchy, InvGamma



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
                 seed=None,
                 estimator='flipout',
                 transform=lambda x: x,
                 inv_transform=lambda x: x):
        """Construct an array of Parameter(s)."""

        # Check types
        assert isinstance(shape, (int, list, np.ndarray)), \
            ('shape must be an int, list of ints, or a numpy ndarray')
        if isinstance(shape, int):
            assert shape > 0, 'shape must be positive'
        if isinstance(shape, list):
            for t_shape in shape:
                assert isinstance(t_shape, int), 'shape must be integer(s)'
        if isinstance(shape, np.ndarray):
            assert shape.dtype.char in np.typecodes['AllInteger'], \
                'shape must be integer(s)'
            assert np.all(shape >= 0), 'shape must be positive'
        assert isinstance(name, str), 'name must be a string'
        assert prior is None or isinstance(prior, BaseDistribution), \
            'prior must be a probflow distribution or None'
        assert issubclass(posterior_fn, BaseDistribution), \
            'posterior_fn must be a probflow distribution'
        assert estimator is None or isinstance(estimator, str), \
            'estimator must be None or a string'

        # Check for valid posterior if using flipout
        sym_dists = [Normal, StudentT, Cauchy]
        if estimator == 'flipout' and posterior_fn not in sym_dists:
            raise ValueError('flipout requires a symmetric posterior ' +
                             'distribution in the location-scale family')

        # If shape is an integer, make it a list
        if isinstance(shape, int):
            shape = [shape]

        # If shape is a numpy vector, make it a list
        # TODO

        # Assign attributes
        self.shape = shape
        self.name = name
        self.prior = prior
        self.posterior_fn = posterior_fn
        self.seed = seed
        self.estimator = estimator
        self.transform = transform
        self.inv_transform = inv_transform
        self._built_posterior = None
        self._session = None
        self._is_built = False

        # TODO: initializer?


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


    def build(self, data, batch_shape):
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
        # TODO: getting an error if you try to make duplicate *non-default* names
        new_name = self.name
        ix = 1
        while tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=new_name):
            new_name = self.name + '_' + str(ix)
            ix += 1
        self.name = new_name


    def _build_prior(self, data, batch_shape):
        """Build the parameter's prior distribution."""
        if self.prior is not None:
            self.prior.build(data, batch_shape)
            self._built_prior = self.prior.built_obj
            # TODO: Check that the built prior shape is broadcastable w/ self.shape


    def _build_posterior(self, data, batch_shape):
        """Build the parameter's posterior distribution."""

        # Create posterior distribution parameters
        params = dict()
        with tf.variable_scope(self.name):
            for arg in self.posterior_fn._post_param_bounds:
                params[arg] = tf.get_variable(arg, shape=self.shape)

        # Transform posterior parameters
        for arg in self.posterior_fn._post_param_bounds:
            lb = self.posterior_fn._post_param_bounds[arg][0]
            ub = self.posterior_fn._post_param_bounds[arg][1]
            params[arg] = self._bound(params[arg], lb, ub)

        # Create variational posterior distribution
        self._params = params
        self.posterior = self.posterior_fn(**params)
        self.posterior.build(data, batch_shape)
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
        if self.estimator is None:
            samples = self._built_posterior.sample(sample_shape=batch_shape,
                                                   seed=seed_stream())

        # Use the Flipout estimator (https://arxiv.org/abs/1803.04386)
        # TODO: this isn't actually the Flipout estimator...
        # it flips samples around the posterior mean but not in the same way...
        elif self.estimator == 'flipout':

            # Create a centered version of the posterior
            params = self._params.copy()
            params['loc'] = 0.0
            sample_posterior = self.posterior_fn(**params)
            sample_posterior.build(data, batch_shape)

            # Take a single sample from that centered posterior
            sample = sample_posterior.built_obj.sample(seed=seed_stream())

            # Generate random sign matrix
            signs = random_rademacher(tf.concat([batch_shape, self.shape], 0),
                                      dtype=data.dtype, seed=seed_stream())

            # Flipout(ish)-generated samples
            samples = sample*signs + self._mean_obj_raw

        # No other estimators supported at the moment
        else:
            raise ValueError('estimator must be None or \'flipout\'')

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


    def plot_posterior(self, num_samples=1000, style='fill', bins=20, ci=0.95,
                       bw=0.075, alpha=0.4):
        """Plot distribution of samples from the posterior distribution.

        TODO: this is similar to _sample(), but returns a numpy array
        (meant to be used by the user to examine the posterior dist)

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
            ``kde=False``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.95
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        """

        def approx_kde(data, bins=500, bw=0.075):
            """A fast approximation to kernel density estimation."""
            stds = 3 #use a gaussian kernel w/ this many std devs
            counts, be = np.histogram(data, bins=bins)
            db = be[1]-be[0]
            pad = 0.5*bins*bw*stds*db
            pbe = np.arange(db, pad, db)
            x_out = np.concatenate((be[0]-np.flip(pbe),
                                   be[0:-1] + np.diff(be),
                                   be[-1]+pbe))
            z_pad = np.zeros(pbe.shape[0])
            raw = np.concatenate((z_pad, counts, z_pad))
            k_x = np.linspace(-stds, stds, bins*bw*stds)
            kernel = 1.0/np.sqrt(2.0*np.pi)*np.exp(-np.square(k_x)/2.0)
            y_out = np.convolve(raw, kernel, mode='same')
            return x_out, y_out

        # Sample from the posterior
        Np = np.prod(self.shape) #number of parameters
        samples = self.sample_posterior(num_samples=num_samples)

        # Compute confidence intervals
        if ci:
            cis = np.empty((Np, 2))
            ci0 = 100 * (0.5 - ci/2.0);
            ci1 = 100 * (0.5 + ci/2.0);
            for i in range(Np):
                cis[i,:] = np.percentile(samples[:,i], [ci0, ci1])

        # Plot the samples
        if style == 'line':
            for i in range(Np):
                px, py = approx_kde(samples[:,i], bw=bw)
                p1 = plt.plot(px, py)
                if ci:
                    yci = np.interp(cis[i,:], px, py)
                    plt.plot([cis[i,0], cis[i,0]], [0, yci[0]], 
                             ':', color=p1[0].get_color())
                    plt.plot([cis[i,1], cis[i,1]], [0, yci[1]], 
                             ':', color=p1[0].get_color())
        elif style == 'fill':
            for i in range(Np):
                color = next(plt.gca()
                             ._get_patches_for_fill
                             .prop_cycler)['color']
                px, py = approx_kde(samples[:,i], bw=bw)
                p1 = plt.fill(px, py, facecolor=color, alpha=alpha)
                if ci:
                    k = (px>cis[i,0]) & (px<cis[i,1])
                    kx = px[k]
                    ky = py[k]
                    plt.fill(np.concatenate(([kx[0]], kx, [kx[-1]])),
                             np.concatenate(([0], ky, [0])),
                             facecolor=color, alpha=alpha)
        elif style == 'hist':
            for i in range(Np):
                _, be, patches = plt.hist(samples[:,i], alpha=alpha, bins=bins)
                if ci:
                    k = (samples[:,i]>cis[i,0]) & (samples[:,i]<cis[i,1])
                    plt.hist(samples[k,i], alpha=alpha, bins=be, 
                             color=patches[0].get_facecolor())

        # Label with parameter name, and no y axis needed
        plt.xlabel(self.name)
        plt.gca().get_yaxis().set_visible(False)


    def __str__(self, prepend=''):
        """String representation of a parameter."""
        return 'Parameter \''+self.name+'\''



class ScaleParameter(Parameter):
    r"""Standard deviation parameter.

    This is a convenience class for creating a standard deviation parameter
    (\sigma).  It is created by first constructing a variance parameter
    (:math:`\sigma^2`) which uses an inverse gamma distribution as the
    variational posterior.

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
    posterior_fn : |Distribution|
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.InvGamma`
    seed : int, float, or |None|
        Seed for the random number generator.
        Set to |None| to use the global seed.
        Default = |None|


    Examples
    --------
    TODO

    """

    def __init__(self,
                 shape=1,
                 name='ScaleParameter',
                 prior=None,
                 posterior_fn=InvGamma,
                 seed=None):
        super().__init__(shape=shape,
                         name=name,
                         prior=prior,
                         posterior_fn=posterior_fn,
                         seed=seed,
                         estimator=None,
                         transform=lambda x: tf.sqrt(x),
                         inv_transform=lambda x: tf.square(x))



# TODO: add support for discrete Parameters?
# In theory can just set posterior_fn to
# Bernoulli or Categorical, and make mean() return the mode?
# and have n_categories-1 different underlying tf variables
# and transform them according to the additive logistic transformation?
# to get probs of categories
# https://en.wikipedia.org/wiki/Logit-normal_distribution#Probability_density_function_2
