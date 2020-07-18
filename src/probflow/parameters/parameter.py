from typing import Callable, Dict, List, Type, Union

import matplotlib.pyplot as plt
import numpy as np

import probflow.utils.ops as O
from probflow.distributions import Normal
from probflow.utils.base import BaseDistribution, BaseParameter
from probflow.utils.casting import to_numpy
from probflow.utils.initializers import scale_xavier, xavier
from probflow.utils.plotting import plot_dist
from probflow.utils.settings import Sampling, get_backend, get_samples


class Parameter(BaseParameter):
    r"""Probabilistic parameter(s).

    A probabilistic parameter :math:`\beta`.  The default posterior
    distribution is the :class:`.Normal` distribution, and the default prior
    is a :class:`.Normal` distribution with a mean of 0 and a standard
    deviation of 1.

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
    shape : int or List[int]
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
    initializer : Dict[str, callable]
        Initializer functions to use for each variable of the variational
        posterior distribution.  Keys correspond to variable names (arguments
        to the distribution), and values contain functions to initialize those
        variables given ``shape`` as the single argument.
    var_transform : Dict[str, callable]
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

    TODO: creating variables

    TODO: creating variable w/ beta posterior

    TODO: plotting posterior dist

    TODO: using __getitem__

    """

    def __init__(
        self,
        shape: Union[int, List[int]] = 1,
        posterior: Type[BaseDistribution] = Normal,
        prior: BaseDistribution = Normal(0, 1),
        transform: Callable = None,
        initializer: Dict[str, Callable] = {
            "loc": xavier,
            "scale": scale_xavier,
        },
        var_transform: Dict[str, Callable] = {
            "loc": None,
            "scale": O.softplus,
        },
        name: str = "Parameter",
    ):

        # Make shape a list
        if isinstance(shape, int):
            shape = [shape]

        # Check values
        if any(e < 1 for e in shape):
            raise ValueError("all shapes must be >0")

        # Assign attributes
        self.shape = shape
        self.posterior_fn = posterior
        self.prior = prior
        self.transform = transform if transform else lambda x: x
        self.initializer = initializer
        self.name = name
        self.var_transform = {
            n: (f if f else lambda x: x) for (n, f) in var_transform.items()
        }

        # Create variables for the variational distribution
        self.untransformed_variables = dict()
        for var, init in initializer.items():
            if get_backend() == "pytorch":
                import torch

                self.untransformed_variables[var] = torch.nn.Parameter(
                    init(shape)
                )
            else:
                import tensorflow as tf

                self.untransformed_variables[var] = tf.Variable(init(shape))

    @property
    def n_parameters(self):
        """Get the number of independent parameters"""
        return int(np.prod(self.shape))

    @property
    def n_variables(self):
        """Get the number of underlying variables"""
        return int(
            sum(
                [
                    np.prod(e.shape.as_list())
                    for e in self.untransformed_variables.values()
                ]
            )
        )

    @property
    def trainable_variables(self):
        """Get a list of trainable variables from the backend"""
        return [e for e in self.untransformed_variables.values()]

    @property
    def variables(self):
        """Variables after applying their respective transformations"""
        return {
            name: self.var_transform[name](val)
            for name, val in self.untransformed_variables.items()
        }

    @property
    def posterior(self):
        """This Parameter's variational posterior distribution"""
        return self.posterior_fn(**self.variables)

    def __call__(self):
        """Return a sample from or the MAP estimate of this parameter.

        TODO

        Returns
        -------
        sample : Tensor
            A sample from this Parameter's variational posterior distribution
        """
        n_samples = get_samples()
        if n_samples is None:
            return self.transform(self.posterior.mean())
        elif n_samples == 1:
            return self.transform(self.posterior.sample())
        else:
            return self.transform(self.posterior.sample(n_samples))

    def kl_loss(self):
        """Compute the sum of the Kullback–Leibler divergences between this
        parameter's priors and its variational posteriors."""
        if self.prior is None:
            return O.zeros([])
        else:
            return O.sum(
                O.kl_divergence(self.posterior, self.prior), axis=None
            )

    def posterior_mean(self):
        """Get the mean of the posterior distribution(s).

        TODO
        """
        return to_numpy(self())

    def posterior_sample(self, n: int = 1):
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
        if n < 1:
            raise ValueError("n must be positive")
        with Sampling(n=n):
            return to_numpy(self())

    def prior_sample(self, n: int = 1):
        """Sample from the prior distribution.


        Parameters
        ----------
        n : int > 0
            Number of samples to draw from the prior distribution.
            Default = 1


        Returns
        -------
        |ndarray|
            Samples from the parameter prior distribution.  If ``n>1`` of size
            ``(num_samples, self.prior.shape)``.  If ``n==1``, of size
            ``(self.prior.shape)``.
        """
        if self.prior is None:
            return np.full(n, np.nan)
        elif n == 1:
            return to_numpy(self.transform(self.prior.sample()))
        else:
            return to_numpy(self.transform(self.prior.sample(n)))

    def posterior_ci(self, ci: float = 0.95, n: int = 10000):
        """Posterior confidence intervals

        Parameters
        ----------
        ci : float
            Confidence interval for which to compute the upper and lower
            bounds.  Must be between 0 and 1.
            Default = 0.95
        n : int
            Number of samples to draw from the posterior distributions for
            computing the confidence intervals
            Default = 10,000

        Returns
        -------
        lb : float or |ndarray|
            Lower bound of the confidence interval
        ub : float or |ndarray|
            Upper bound of the confidence interval
        """

        # Check values
        if ci < 0.0 or ci > 1.0:
            raise ValueError("ci must be between 0 and 1")

        # Sample from the posterior
        samples = self.posterior_sample(n=n)

        # Compute confidence intervals
        ci0 = 100 * (0.5 - ci / 2.0)
        ci1 = 100 * (0.5 + ci / 2.0)
        bounds = np.percentile(samples, q=[ci0, ci1], axis=0)
        return bounds[0, ...], bounds[1, ...]

    def posterior_plot(
        self,
        n: int = 10000,
        style: str = "fill",
        bins: Union[int, list, np.ndarray] = 20,
        ci: float = 0.0,
        bw: float = 0.075,
        alpha: float = 0.4,
        color=None,
        **kwargs
    ):
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
        kwargs
            Additional keyword arguments are passed to
            :meth:`.utils.plotting.plot_dist`
        """

        # Sample from the posterior
        samples = self.posterior_sample(n=n)

        # Plot the posterior densities
        plot_dist(
            samples,
            xlabel=self.name,
            style=style,
            bins=bins,
            ci=ci,
            bw=bw,
            alpha=alpha,
            color=color,
            **kwargs
        )

        # Label with parameter name
        plt.xlabel(self.name)

    def prior_plot(
        self,
        n: int = 10000,
        style: str = "fill",
        bins: Union[int, list, np.ndarray] = 20,
        ci: float = 0.0,
        bw: float = 0.075,
        alpha: float = 0.4,
        color=None,
    ):
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

        # Sample from the posterior
        samples = self.prior_sample(n=n)

        # Plot the posterior densities
        plot_dist(
            samples,
            xlabel=self.name,
            style=style,
            bins=bins,
            ci=ci,
            bw=bw,
            alpha=alpha,
            color=color,
        )

        # Label with parameter name
        plt.xlabel(self.name + " prior")

    def _get_one_dim(self, val, key, axis):
        """Slice along one axis, keeping the dimensionality of the input"""
        if isinstance(key, slice):
            if any(k is not None for k in [key.start, key.stop, key.step]):
                ix = np.arange(*key.indices(val.shape[axis]))
                return O.gather(val, ix, axis=axis)
            else:
                return val
        elif isinstance(key, int):
            key %= val.shape[axis]
            return O.gather(val, [key], axis=axis)
        else:
            return O.gather(val, key, axis=axis)

    def __getitem__(self, key):
        """Get a slice of a sample from the parameter"""
        x = self()
        if isinstance(key, tuple):
            iA = 0
            for i in range(len(key)):
                if key[i] is Ellipsis:
                    iA = x.ndim - len(key) + i
                else:
                    x = self._get_one_dim(x, key[i], iA)
                iA += 1
            return x
        elif key is Ellipsis:
            return x
        else:
            return self._get_one_dim(x, key, 0)

    def __repr__(self):
        return (
            "<pf."
            + self.__class__.__name__
            + " "
            + self.name
            + " shape="
            + str(self.shape)
            + ">"
        )
