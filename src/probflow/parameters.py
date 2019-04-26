"""Parameters.

Parameters are values which characterize the behavior of a model.  When
fitting a model, we want to find the values of the parameters which
best allow the model to explain the data.  However, with Bayesian modeling
we want not only to find the single *best* value for each parameter, but a 
probability distribution which describes how likely any given value of 
a parameter is to be the best or true value.

Parameters have both priors (probability distributions which describe how
likely we think different values for the parameter are *before* taking into
consideration the current data), and posteriors (probability distributions 
which describe how likely we think different values for the parameter are
*after* taking into consideration the current data).  The prior is set 
to a specific distribution before fitting the model.  While the *type* of 
distribution used for the posterior is set before fitting the model, the 
shape of that distribution is determined while fitting the model.
See the :ref:`math` section for more info.

The :class:`.Parameter` class can be used to create any probabilistic
parameter. 

For convenience, ProbFlow also includes some classes which are special cases
of a :class:`.Parameter`:

* :class:`.ScaleParameter` - standard deviation parameter
* :class:`.CategoricalParameter` - categorical parameter

----------

"""

__all__ = [
    'Parameter',
    'ScaleParameter',
    'CategoricalParameter',
]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow_probability.python.math import random_rademacher

from .core import BaseParameter, BaseDistribution
from .distributions import Normal, StudentT, Cauchy, InvGamma, Categorical
from .utils.plotting import plot_dist, centered_text


class Parameter(BaseParameter):
    r"""Parameter(s) drawn from variational distribution(s).

    A probabilistic parameter $\beta$.  The default posterior distribution
    is the Normal distribution, and the default prior is a Normal 
    distribution with a mean of 0 and a standard deviation of 1.

    The prior for a |Parameter| can be set to any |Distribution| object
    (via the ``prior`` argument), and the type of distribution to use for the
    posterior can be set to any |Distribution| class (using the ``posterior``
    argument).

    The parameter can be given a specific name using the ``name`` argument.
    This makes it easier to access specific parameters after fitting the
    model (e.g. in order to view the posterior distribution).

    The number of independent parameters represented by this 
    :class:`.Parameter` object can be set using the ``shape`` argument.  For 
    example, to create a vector of 5 parameters, set ``shape=5``, or to create
    a 20x7 matrix of parameters set ``shape=[20,7]``.


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

    Create a scalar parameter which represents the slope of a line::

        from probflow import Parameter, Input, Normal

        slope = Parameter()
        feature = Input()
        model = Normal(slope*feature + 3, 1.0)

    Create a vector of parameters which represent coefficients for each 
    feature dimension::

        from probflow import Parameter, Input, Normal

        weights = Parameter(shape=3)
        features = Input([0, 1, 2])
        model = Normal(Dot(weights, features) + 3, 1.0)

    Create a parameter which has a Cauchy prior and posterior, instead of the
    default Normal::

        from probflow import Parameter, Cauchy, Input, Normal

        weight = Parameter(prior=Cauchy(0, 1),
                           posterior=Cauchy)
        feature = Input()
        model = Normal(weight*feature + 3, 1.0)

    View the prior distribution which was used for a parameter::

        weight.prior_plot()

    View the posterior distribution for the parameter after fitting the
    model::

        # x and y are Numpy arrays or pandas DataFrame/Series
        model.fit(x, y)

        weight.posterior_plot()

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

        Bound distribution arguments by applying a transformation: an 
        exponential transformation when there is a bound on one side, or a 
        sigmoid transformation when both sides are bounded.

        TODO: just use tf constraints

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
        """Build the parameter and all elements of its priors and posteriors.

        Parameters
        ----------
        data : |Tensor|
            Data for this batch.
        batch_shape : |Tensor|
            Batch shape.
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
                                                  initializer=init,
                                                  dtype=data.dtype)
                elif isinstance(self.initializer, dict):
                    params[arg] = \
                        tf.get_variable(arg, dtype=data.dtype,
                                        initializer=self.initializer[arg])
                else:
                    params[arg] = \
                        tf.get_variable(arg, dtype=data.dtype,
                                        initializer=self.initializer)

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
        try:
            built_mean = self._built_posterior.mean()
        except NotImplementedError:
            built_mean = self._built_posterior.mode()
        self._mean_obj_raw = tf.expand_dims(built_mean, 0)
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

        .. admonition:: Model must be fit first!

            Before calling :meth:`.posterior_mean` on a |Parameter|, you must
            first :meth:`fit <.BaseDistribution.fit>` the model to which it
            belongs to some data.

        Returns
        -------
        |ndarray|
            Mean of the parameter posterior distribution.  
            Size ``self.shape``.
        """
        self._ensure_is_built()
        self._ensure_is_fit()
        try:
            mean_op = self._built_posterior.mean()
        except NotImplementedError:
            mean_op = self._built_posterior.mode()
        mean_op = self.transform(mean_op)
        mean = self._session.run(mean_op)
        return mean


    def posterior_sample(self, num_samples=1000):
        """Sample from the posterior distribution.

        .. admonition:: Model must be fit first!

            Before calling :meth:`.posterior_sample` on a |Parameter|, you
            must first :meth:`fit <.BaseDistribution.fit>` the model to which
            it belongs to some data.

        Parameters
        ----------
        num_samples : int > 0
            Number of samples to draw from the posterior distribution.
            Default = 1000

        Returns
        -------
        |ndarray|
            Samples from the parameter posterior distribution.  Of size
            ``(num_samples, self.shape)``.
        """

        # Check num_samples
        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an int')
        if num_samples < 1:
            raise ValueError('num_samples must be positive')

        # Ensure model is fit
        self._ensure_is_built()
        self._ensure_is_fit()

        # Return the samples
        samples_op = self._built_posterior.sample(sample_shape=num_samples)
        samples_op = self.transform(samples_op)
        samples = self._session.run(samples_op)
        return samples


    def prior_sample(self, num_samples=1000):
        """Sample from the prior distribution.

        .. admonition:: Model must be fit first!

            Before calling :meth:`.prior_sample` on a |Parameter|, you must
            first :meth:`fit <.BaseDistribution.fit>` the model to which it
            belongs to some data.

        Parameters
        ----------
        num_samples : int > 0
            Number of samples to draw from the posterior distribution.
            Default = 1000

        Returns
        -------
        |ndarray|
            Samples from the parameter prior distribution.  Of size
            ``(num_samples,self.shape)``.  If this parameter has not prior, 
            returns an empty list.
        """

        # Check num_samples
        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an int')
        if num_samples < 1:
            raise ValueError('num_samples must be positive')

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


    def posterior_plot(self, num_samples=1000, style='fill', bins=20, ci=0.0,
                       bw=0.075, alpha=0.4, color=None):
        """Plot distribution of samples from the posterior distribution.

        .. admonition:: Model must be fit first!

            Before calling :meth:`.posterior_plot` on a |Parameter|, you
            must first :meth:`fit <.BaseDistribution.fit>` the model to which
            it belongs to some data.

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
        alpha : float between 0 and 1
            Transparency of fill/histogram
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        """

        # Check inputs
        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an int')
        if num_samples < 1:
            raise ValueError('num_samples must be positive')
        if type(style) is not str or style not in ['fill', 'line', 'hist']:
            raise TypeError("style must be \'fill\', \'line\', or \'hist\'")
        if not isinstance(bins, (int, float, np.ndarray)):
            raise TypeError('bins must be an int or list or numpy vector')
        if type(ci) is not float or ci<0.0 or ci>1.0:
            raise TypeError('ci must be a float between 0 and 1')
        if type(alpha) is not float or alpha<0.0 or alpha>1.0:
            raise TypeError('alpha must be a float between 0 and 1')

        # Sample from the posterior
        samples = self.posterior_sample(num_samples=num_samples)
        
        # Plot the posterior densities
        plot_dist(samples, xlabel=self.name, style=style, bins=bins, 
                  ci=ci, bw=bw, alpha=alpha, color=color)


    def prior_plot(self, num_samples=1000, style='fill', bins=20, ci=0.0,
                   bw=0.075, alpha=0.4, color=None):
        """Plot distribution of samples from the prior distribution.

        .. admonition:: Model must be fit first!

            Before calling :meth:`.prior_plot` on a |Parameter|, you
            must first :meth:`fit <.BaseDistribution.fit>` the model to which
            it belongs to some data.

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
        alpha : float between 0 and 1
            Transparency of fill/histogram
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
        samples = self.prior_sample(num_samples=num_samples)
        
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

    Use :class:`.ScaleParameter` to create a standard deviation parameter
    for a :class:`.Normal` distribution::

        from probflow import ScaleParameter, Normal

        std_dev = ScaleParameter()
        model = Normal(0.0, std_dev)
        model.fit(x, y)
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



class CategoricalParameter(Parameter):
    r"""Categorical parameter.

    This is a convenience class for creating a categorical parameter.
    It is created by first constructing :math:`N-1` variables :math:`\theta_j` 
    for :math:`j \in {1,...,N-1}`.  These variables are transformed into
    :math:`N` category probabilities :math:`p_i` for :math:`i \in {1,...,N}`
    using the additive logistic transformation:

    .. math::

        p_i = \frac{\exp \theta_i}{1+\sum_{j=1}^{N-1} \exp \theta_j}
        ~ \text{for} ~ i \in \{ 1, ..., N-1 \}

    and

    .. math::

        p_N = \frac{1}{1+\sum_{j=1}^{N-1} \exp \theta_j}

    By default, a uniform prior is used.

    The category values can be set using the ``values`` keyword argument.
    By default, the emitted category values are integers starting at 0.


    Parameters
    ----------
    values : int or list of float or 1D |ndarray|
        Values corresponding to each category, or the number of unique values.
        If an integer, parameter has ``values`` categories, and category
        values are integers starting at 0.  I.e., the first category has value
        0, the second category has value 1, etc.  If ``values`` is a list or
        an |ndarray|, these are the values corresponding to each category,
        such that there are ``len(values)`` unique categories.
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

    Create a :class:`.CategoricalParameter` with 5 unique categories::

        from probflow import CategoricalParameter

        theta = CategoricalParameter(5)

    Use :class:`.CategoricalParameter` to create a parameter which only takes
    values of -1, 0, or 1::

        from probflow import CategoricalParameter, Normal

        theta = CategoricalParameter([-1, 0, 1])

    Use :class:`.CategoricalParameter` to create a 10-by-3 array of 
    parameters, each of which can take one of 5 unique categories::

        from probflow import CategoricalParameter

        theta = CategoricalParameter(5, shape=[10, 3])
    """

    def __init__(self, values,
                 shape=1,
                 name='CategoricalParameter',
                 prior=None,
                 posterior=Categorical,
                 seed=None,
                 initializer=None):

        # Make shape a list
        if isinstance(shape, int):
            shape = [shape]

        # Create ``values`` unique categories
        if isinstance(values, int):
            Nc = values
            transform = lambda x: x
            inv_transform = lambda x: x

        # Create ``len(values)`` categories with specific output values
        elif isinstance(values, (list, np.ndarray)):
            Nc = len(values)
            transform = lambda x: tf.gather(values, x)
            table = tf.contrib.lookup.HashTable( #inverse transform w/ lookup
                tf.contrib.lookup.KeyValueTensorInitializer(
                    values, np.arange(Nc)), values[0])
            inv_transform = lambda x: table.lookup(x)

        else:
            raise TypeError('values must be an int, list, or ndarray')

        # Set uniform prior if none passed
        if prior is None:
            # TODO: won't work if data to fit isn't float32
            # need to somehow dynamically cast the logits correctly...
            prior = Categorical(np.full(shape+[Nc], 1.0/Nc).astype('float32'),
                                input_type='probs')

        # Posterior logits include each class
        shape = shape+[Nc-1]

        # Call Parameter's init
        super().__init__(shape=shape,
                         name=name,
                         prior=prior,
                         posterior=posterior,
                         seed=seed,
                         transform=transform,
                         inv_transform=inv_transform,
                         initializer=initializer)



# TODO: add support for discrete Parameters?
# In theory can just set posterior to
# Bernoulli or Categorical, and make mean() return the mode?
# and have n_categories-1 different underlying tf variables
# and transform them according to the additive logistic transformation?
# to get probs of categories
# https://en.wikipedia.org/wiki/Logit-normal_distribution#Probability_density_function_2

# TODO: DeterministicParameter
# no distribution, just a single value (using Deterministic distribution)
# just a convenience so user doesn't have to manually set the posterior
# to Deterministic
