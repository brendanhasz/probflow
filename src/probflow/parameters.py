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
shape of that distribution (the value of the parameters which define the
distribution) is optimized while fitting the model.
See the :ref:`math` section for more info.

The :class:`.Parameter` class can be used to create any probabilistic
parameter. 

For convenience, ProbFlow also includes some classes which are special cases
of a :class:`.Parameter`:

* :class:`.ScaleParameter` - standard deviation parameter
* :class:`.CategoricalParameter` - categorical parameter
* :class:`.BoundedParameter` - parameter which is bounded between 0 and 1
* :class:`.PositiveParameter` - parameter which is always greater than 0

----------

"""

__all__ = [
    'Parameter',
    'ScaleParameter',
    'CategoricalParameter',
    'BoundedParameter',
    'PositiveParameter',
]


import probflow.core.ops as O
from probflow.core.settings import get_sampling
from probflow.core.settings import get_backend
from probflow.plotting import plot_dist
from probflow.utils.initializers import xavier
from probflow.utils.initializers import scale_xavier
from probflow.utils.initializers import pos_xavier



class Parameter(BaseParameter):
    r"""Probabilistic parameter(s).

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
    shape : int, list of int, or |ndarray|
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0,1)``
    transform : callable
        Transform to apply to the random variable.  For example, to create a
        parameter with an inverse gamma posterior, use
        ``posterior``=:class:`.Gamma`` and
        ``transform = lambda x: tf.reciprocal(x)``
        Default is to use no transform.
    initializer : dict of callables
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : dict of callables
        Transform to apply to each variable of the variational posterior.
        For example to transform the standard deviation parameter from 
        untransformed space to transformed, positive, space, use
        ``initializer={'scale': tf.random.randn}`` and
        ``var_transform={'scale': tf.nn.softplus}``
    name : str
        Name of the parameter(s).
        Default = ``'Parameter'``


    Examples
    --------

    TODO: creating variable

    TODO: creating variable w/ beta posterior

    TODO: plotting posterior dist

    """

    def __init__(self,
                 shape=1,
                 posterior=Normal,
                 prior=Normal(0, 1),
                 transform=lambda x: x,
                 initializer={'loc': xavier, 'scale': scale_xavier},
                 var_transform={'loc': lambda x: x, 'scale': O.softplus},
                 name='Parameter'):
        """Construct an array of Parameter(s)."""

        # Check types
        # TODO

        # Make shape a list
        if isinstance(shape, tuple):
            shape = list(shape)
        if isinstance(shape, int):
            shape = [shape]
        if isinstance(shape, np.ndarray):
            shape = shape.tolist()

        # Assign attributes
        self.shape = shape
        self.posterior = posterior
        self.prior = prior
        self.transform = transform
        self.initializer = initializer
        self.var_transform = var_transform
        self.name = name

        # Create variables for the variational distribution
        self.variables = dict()
        for var, init in initializer.items():
            if get_backend() == 'pytorch':
                self.variables[var] = init(shape)
                self.variables[var].requires_grad = True
            else:
                self.variables[var] = tf.Variable(init(shape))


    def _t_vars(self):
        """Variables after applying their respective transformations"""
        return {name: self.var_transform[name](val)
                for name, val in self.variables.items()}


    def __call__(self):
        """Return a sample from or the MAP estimate of this parameter.

        TODO
        """
        if get_sampling():
            return self.transform(self.posterior(**self._t_vars()).sample())
        else:
            return self.transform(self.posterior(**self._t_vars()).mean())


    def kl_loss(self):
        """Compute the sum of the Kullback–Leibler divergences between this
        parameter's priors and its variational posteriors."""
        return O.sum(O.kl_divergence(self.posterior, self.prior))


    def posterior_mean(self):
        """Get the mean of the posterior distribution(s).

        TODO
        """
        return self.transform(self.posterior(**self._t_vars()).mean())


    def posterior_sample(self, n=1):
        """Sample from the posterior distribution.

        Parameters
        ----------
        n : int > 0
            Number of samples to draw from the posterior distribution.
            Default = 1

        Returns
        -------
        TODO
        """
        if n==1:
            return self.transform(self.posterior(**self._t_vars()).sample())
        else:
            return self.transform(self.posterior(**self._t_vars()).sample(n))


    def prior_sample(self, n=1):
        """Sample from the prior distribution.

        .. admonition:: Model must be fit first!

            Before calling :meth:`.prior_sample` on a |Parameter|, you must
            first :meth:`fit <.BaseDistribution.fit>` the model to which it
            belongs to some data.

        Parameters
        ----------
        n : int > 0
            Number of samples to draw from the prior distribution.
            Default = 1

        Returns
        -------
        |ndarray|
            Samples from the parameter prior distribution.  If ``n>1`` of size
            ``(num_samples, self.prior.shape)``.  If ``n==1```, of size
            ``(self.prior.shape)``.
        """
        if n==1:
            return self.transform(self.prior.sample())
        else:
            return self.transform(self.prior.sample(n))


    def posterior_ci(self, ci=0.95, n=10000):
        """Posterior confidence intervals

        Parameters
        ----------
        ci : float
            Confidence interval for which to compute the upper and lower
            bounds.  Must be between 0 and 1.
        n : int
            Number of samples to draw from the posterior distributions for
            computing the confidence intervals

        Returns
        -------
        lb : float or |ndarray|
            Lower bound of the confidence interval
        ub : float or |ndarray|
            Upper bound of the confidence interval
        """

        # Check inputs
        if not isinstance(ci, float):
            raise TypeError('ci must be a float')
        if ci<0.0 or ci>1.0:
            raise ValueError('ci must be between 0 and 1')
        if not isinstance(n, int):
            raise TypeError('n must be an int')
        if n < 1:
            raise ValueError('n must be positive')

        # Sample from the posterior
        samples = self.posterior_sample(n=n)

        # Compute confidence intervals
        ci0 = 100 * (0.5 - ci/2.0)
        ci1 = 100 * (0.5 + ci/2.0)
        bounds = np.percentile(samples, q=[ci0, ci1], axis=0)
        return bounds[0, ...], bounds[1, ...]


    def posterior_plot(self, n=10000, style='fill', bins=20, ci=0.0,
                       bw=0.075, alpha=0.4, color=None):
        """Plot distribution of samples from the posterior distribution.

        Parameters
        ----------
        n : int
            Number of samples to take from each posterior distribution for
            estimating the density.  Default = 10000
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
        if not isinstance(n, int):
            raise TypeError('n must be an int')
        if n < 1:
            raise ValueError('n must be positive')
        if type(style) is not str or style not in ['fill', 'line', 'hist']:
            raise TypeError("style must be \'fill\', \'line\', or \'hist\'")
        if not isinstance(bins, (int, float, np.ndarray)):
            raise TypeError('bins must be an int or list or numpy vector')
        if not isinstance(ci, float):
            raise TypeError('ci must be a float')
        if ci<0.0 or ci>1.0:
            raise ValueError('ci must be between 0 and 1')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha<0.0 or alpha>1.0:
            raise TypeError('alpha must be between 0 and 1')

        # Sample from the posterior
        samples = self.posterior_sample(n=n)
        
        # Plot the posterior densities
        plot_dist(samples, xlabel=self.name, style=style, bins=bins, 
                  ci=ci, bw=bw, alpha=alpha, color=color)


    def prior_plot(self, n=10000, style='fill', bins=20, ci=0.0,
                   bw=0.075, alpha=0.4, color=None):
        """Plot distribution of samples from the prior distribution.

        Parameters
        ----------
        n : int
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

        # Check inputs
        if not isinstance(n, int):
            raise TypeError('n must be an int')
        if n < 1:
            raise ValueError('n must be positive')
        if type(style) is not str or style not in ['fill', 'line', 'hist']:
            raise TypeError("style must be \'fill\', \'line\', or \'hist\'")
        if not isinstance(bins, (int, float, np.ndarray)):
            raise TypeError('bins must be an int or list or numpy vector')
        if not isinstance(ci, float):
            raise TypeError('ci must be a float')
        if ci<0.0 or ci>1.0:
            raise ValueError('ci must be between 0 and 1')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha<0.0 or alpha>1.0:
            raise TypeError('alpha must be between 0 and 1')

        # Sample from the posterior
        samples = self.prior_sample(n=n)
        
        # Plot the posterior densities
        plot_dist(samples, xlabel=self.name, style=style, bins=bins, 
                  ci=ci, bw=bw, alpha=alpha, color=color)



class ScaleParameter(Parameter):
    r"""Standard deviation parameter.

    This is a convenience class for creating a standard deviation parameter
    (:math:`\sigma`).  It is created by first constructing a variance 
    parameter (:math:`\sigma^2`) which uses an inverse gamma distribution as
    the variational posterior.

    .. math::

        \sigma^2 \sim \text{InverseGamma}(\alpha, \beta)

    Then the variance is transformed into the standard deviation:

    .. math::

        \sigma = \sqrt{\sigma^2}

    By default, an inverse gamma prior is used:

        \sigma^2 \sim \text{InverseGamma}(5, 5)


    Parameters
    ----------
    shape : int, list of int, or |ndarray|
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.InverseGamma`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.InverseGamma` ``(5, 5)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use a square root transform.
    initializer : dict of callables
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : dict of callables
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'ScaleParameter'``

    Examples
    --------

    Use :class:`.ScaleParameter` to create a standard deviation parameter
    for a :class:`.Normal` distribution::

    TODO

    """

    def __init__(self,
                 shape=1,
                 posterior=InverseGamma,
                 prior=InverseGamma(5, 5),
                 transform=lambda x: O.sqrt(x),
                 initializer={'concentration': pos_xavier, 
                              'scale': pos_xavier},
                 var_transform={'concentration': O.softplus,
                                'scale': O.softplus},
                 name='ScaleParameter'):
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)



class CategoricalParameter(Parameter):
    r"""Categorical parameter.

    This is a convenience class for creating a categorical parameter 
    :math:`\theta` with a Dirichlet posterior:

    .. math::

        \theta \sim \text{Dirichlet}(\mathbf{\alpha})

    By default, a uniform Dirichlet prior is used:

        \theta \sim \text{Dirichlet}(\mathbf{1})


    Parameters
    ----------
    k : int > 2
        Number of categories.
    shape : int, list of int, or |ndarray|
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Dirichlet`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Dirichlet` ``(1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use no transform.
    initializer : dict of callables
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : dict of callables
        Transform to apply to each variable of the variational posterior.
    name : str
        Name of the parameter(s).
        Default = ``'Parameter'``


    Examples
    --------

    TODO: creating variable

    """

    def __init__(self,
                 k=2,
                 shape=1,
                 posterior=Dirichlet,
                 prior=None,
                 transform=lambda x: x,
                 initializer={'concentration': pos_xavier, 
                              'scale': pos_xavier},
                 var_transform={'concentration': O.softplus,
                                'scale': O.softplus},
                 name='CategoricalParameter'):

        # Check type of k
        if not isinstance(k, int):
            raise TypeError('k must be an integer')
        if k<2:
            raise ValueError('k must be >1')

        # Make shape a list
        if isinstance(shape, tuple):
            shape = list(shape)
        if isinstance(shape, int):
            shape = [shape]
        if isinstance(shape, np.ndarray):
            shape = shape.tolist()

        # Create shape of underlying variable array
        shape = shape+[k]

        # Use a uniform prior
        if prior is None:
            prior = Dirichlet(O.ones(shape))

        # Initialize the parameter
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)



class BoundedParameter(Parameter):
    r"""A parameter bounded on either side

    This is a convenience class for creating a parameter :math:`\beta` bounded
    on both sides.  It uses a logit-normal posterior distribution:

    .. math::

        \text{Logit}(\beta) \sim \text{Normal}(\mu, \sigma)


    Parameters
    ----------
    shape : int, list of int, or |ndarray|
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0, 1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use a sigmoid transform.
    initializer : dict of callables
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : dict of callables
        Transform to apply to each variable of the variational posterior.
    min : float
        Minimum value the parameter can take.
        Default = 0.
    max : float
        Maximum value the parameter can take.
        Default = 1.
    name : str
        Name of the parameter(s).
        Default = ``'ScaleParameter'``

    Examples
    --------

    TODO

    """

    def __init__(self,
                 shape=1,
                 posterior=Normal,
                 prior=Normal(0, 1),
                 transform=None,
                 initializer={'loc': xavier, 'scale': scale_xavier},
                 var_transform={'loc': lambda x: x, 'scale': O.softplus},
                 min=0.0,
                 max=1.0,
                 name='BoundedParameter'):

        # Create the transform based on the bounds
        if transform is None:
            transform = lambda x: min + (max-min)*O.sigmoid(x)

        # Create the parameter
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)



class PositiveParameter(Parameter):
    r"""A parameter which takes only positive values.

    This is a convenience class for creating a parameter :math:`\beta` which 
    can only take positive values.  It uses a log-normal variational posterior
    distribution:

    .. math::

        \text{Log}(\beta) \sim \text{Normal}(\mu, \sigma)


    Parameters
    ----------
    shape : int, list of int, or |ndarray|
        Shape of the array containing the parameters.
        Default = ``1``
    posterior : |Distribution| class
        Probability distribution class to use to approximate the posterior.
        Default = :class:`.Normal`
    prior : |Distribution| object
        Prior probability distribution function which has been instantiated
        with parameters.
        Default = :class:`.Normal` ``(0, 1)``
    transform : callable
        Transform to apply to the random variable.
        Default is to use an exponential transform.
    initializer : dict of callables
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : dict of callables
        Transform to apply to each variable of the variational posterior.
    min : float
        Minimum value the parameter can take.
        Default = 0.
    max : float
        Maximum value the parameter can take.
        Default = 1.
    name : str
        Name of the parameter(s).
        Default = ``'ScaleParameter'``

    Examples
    --------

    TODO

    """

    def __init__(self,
                 shape=1,
                 posterior=Normal,
                 prior=Normal(0, 1),
                 transform=O.exp,
                 initializer={'loc': xavier, 'scale': scale_xavier},
                 var_transform={'loc': lambda x: x, 'scale': O.softplus},
                 name='PositiveParameter'):
        super().__init__(shape=shape,
                         posterior=posterior,
                         prior=prior,
                         transform=transform,
                         initializer=initializer,
                         var_transform=var_transform,
                         name=name)
