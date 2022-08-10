import matplotlib.pyplot as plt
import numpy as np

from probflow.data import DataGenerator, make_generator
from probflow.utils.casting import to_numpy
from probflow.utils.plotting import plot_by, plot_dist

from .model import Model


class ContinuousModel(Model):
    """Abstract base class for probflow models where the dependent variable
    (the target) is continuous and 1-dimensional.

    The only advantage to using this over the more general :class:`.Model` is
    that :class:`ContinuousModel` also includes several methods specific to
    continuous models, for tasks such as getting the predictive intervals,
    coverage, R-squared value, or calibration metrics (see below for the full
    list of methods).

    .. admonition:: Only supports scalar dependent variables

        Note that the methods of :class:`.ContinuousModel` only support scalar,
        continuous dependent variables (not *any* continuous model, as the name
        might suggest).  For models which have a multidimensional output, just
        use the more general :class:`.Model`; for models with categorical
        output (i.e., classifiers), use :class:`.CategoricalModel`; and for
        models which have a discrete output (e.g. a Poisson regression), use
        :class:`.DiscreteModel`.

    This class inherits several methods from :class:`.Module`:

    * :attr:`~parameters`
    * :attr:`~modules`
    * :attr:`~trainable_variables`
    * :meth:`~kl_loss`
    * :meth:`~kl_loss_batch`
    * :meth:`~reset_kl_loss`
    * :meth:`~add_kl_loss`

    as well as several methods from :class:`.Model`:

    * :meth:`~log_likelihood`
    * :meth:`~train_step`
    * :meth:`~fit`
    * :meth:`~stop_training`
    * :meth:`~set_learning_rate`
    * :meth:`~predictive_sample`
    * :meth:`~aleatoric_sample`
    * :meth:`~epistemic_sample`
    * :meth:`~predict`
    * :meth:`~metric`
    * :meth:`~posterior_mean`
    * :meth:`~posterior_sample`
    * :meth:`~posterior_ci`
    * :meth:`~prior_sample`
    * :meth:`~posterior_plot`
    * :meth:`~prior_plot`
    * :meth:`~log_prob`
    * :meth:`~prob`
    * :meth:`~save`
    * :meth:`~summary`

    and adds the following continuous-model-specific methods:

    * :meth:`~predictive_interval`
    * :meth:`~aleatoric_interval`
    * :meth:`~epistemic_interval`
    * :meth:`~pred_dist_plot`
    * :meth:`~predictive_prc`
    * :meth:`~pred_dist_covered`
    * :meth:`~pred_dist_coverage`
    * :meth:`~coverage_by`
    * :meth:`~r_squared`
    * :meth:`~r_squared_plot`
    * :meth:`~residuals`
    * :meth:`~residuals_plot`
    * :meth:`~calibration_curve`
    * :meth:`~calibration_curve_plot`
    * :meth:`~calibration_metric`
    * :meth:`~sharpness`
    * :meth:`~coefficient_of_variation`


    Example
    -------

    TODO

    """

    def _intervals(self, fn, x, side, ci=0.95, n=1000, batch_size=None):
        """Compute intervals on some type of sample"""

        # Compute in batches?
        if batch_size is not None:
            intervals = [
                self._intervals(fn, x_data, side, ci=ci, n=n)
                for x_data, y_data in make_generator(
                    x, test=True, batch_size=batch_size
                )
            ]
            return (np.concatenate(e, axis=0) for e in zip(*intervals))

        # No batching (or this is a batch)
        samples = fn(x, n=n)
        if side == "lower":
            return np.percentile(samples, 100 * (1.0 - ci), axis=0)
        elif side == "upper":
            return np.percentile(samples, 100 * ci, axis=0)
        else:
            lb = 100 * (1.0 - ci) / 2.0
            prcs = np.percentile(samples, [lb, 100.0 - lb], axis=0)
            return prcs[0, ...], prcs[1, ...]

    def predictive_interval(
        self, x, ci=0.95, side="both", n=1000, batch_size=None
    ):
        r"""Compute confidence intervals on the model's estimate of the target
        given ``x``, including all sources of uncertainty.

        TODO: docs

        TODO: using side= both, upper, vs lower


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        ci : float between 0 and 1
            Inner proportion of predictive distribution to use a the
            confidence interval.
            Default = 0.95
        side : str {'lower', 'upper', 'both'}
            Whether to get the one- or two-sided interval, and which side to
            get.  If ``'both'`` (default), gets the upper and lower bounds of
            the central ``ci`` interval.  If ``'lower'``, gets the lower bound
            on the one-sided ``ci`` interval.  If ``'upper'``, gets the upper
            bound on the one-sided ``ci`` interval.
        n : int
            Number of samples from the posterior predictive distribution to
            take to compute the confidence intervals.
            Default = 1000
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        lb : |ndarray|
            Lower bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='upper'``.
        ub : |ndarray|
            Upper bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='lower'``.
        """
        return self._intervals(
            self.predictive_sample, x, side, ci=ci, n=n, batch_size=batch_size
        )

    def aleatoric_interval(
        self, x, ci=0.95, side="both", n=1000, batch_size=None
    ):
        r"""Compute confidence intervals on the model's estimate of the target
        given ``x``, including only aleatoric uncertainty (uncertainty due to
        noise).

        TODO: docs


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        ci : float between 0 and 1
            Inner proportion of predictive distribution to use a the
            confidence interval.
            Default = 0.95
        side : str {'lower', 'upper', 'both'}
            Whether to get the one- or two-sided interval, and which side to
            get.  If ``'both'`` (default), gets the upper and lower bounds of
            the central ``ci`` interval.  If ``'lower'``, gets the lower bound
            on the one-sided ``ci`` interval.  If ``'upper'``, gets the upper
            bound on the one-sided ``ci`` interval.
        n : int
            Number of samples from the aleatoric predictive distribution to
            take to compute the confidence intervals.
            Default = 1000
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        lb : |ndarray|
            Lower bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='upper'``.
        ub : |ndarray|
            Upper bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='lower'``.
        """
        return self._intervals(
            self.aleatoric_sample, x, side, ci=ci, n=n, batch_size=batch_size
        )

    def epistemic_interval(
        self, x, ci=0.95, side="both", n=1000, batch_size=None
    ):
        r"""Compute confidence intervals on the model's estimate of the target
        given ``x``, including only epistemic uncertainty (uncertainty due to
        uncertainty as to the model's parameter values).

        TODO: docs


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        ci : float between 0 and 1
            Inner proportion of predictive distribution to use a the
            confidence interval.
            Default = 0.95
        side : str {'lower', 'upper', 'both'}
            Whether to get the one- or two-sided interval, and which side to
            get.  If ``'both'`` (default), gets the upper and lower bounds of
            the central ``ci`` interval.  If ``'lower'``, gets the lower bound
            on the one-sided ``ci`` interval.  If ``'upper'``, gets the upper
            bound on the one-sided ``ci`` interval.
        n : int
            Number of samples from the epistemic predictive distribution to
            take to compute the confidence intervals.
            Default = 1000
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        lb : |ndarray|
            Lower bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='upper'``.
        ub : |ndarray|
            Upper bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='lower'``.
        """
        return self._intervals(
            self.epistemic_sample, x, side, ci=ci, n=n, batch_size=batch_size
        )

    def pred_dist_plot(
        self, x, n=10000, cols=1, individually=False, batch_size=None, **kwargs
    ):
        r"""Plot posterior predictive distribution from the model given ``x``.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 10000
        cols : int
            Divide the subplots into a grid with this many columns (if
            ``individually=True``.
        individually : bool
            If ``True``, plot one subplot per datapoint in ``x``, otherwise
            plot all the predictive distributions on the same plot.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).
        **kwargs
            Additional keyword arguments are passed to :func:`.plot_dist`

        Example
        -------

        TODO

        """

        # Sample from the predictive distribution
        samples = self.predictive_sample(x, n=n, batch_size=batch_size)

        # Independent variable must be scalar
        Ns = samples.shape[0]
        N = samples.shape[1]
        if samples.ndim > 2 and any(e > 1 for e in samples.shape[2:]):
            raise NotImplementedError(
                "only scalar dependent variables are supported"
            )
        else:
            samples = samples.reshape([Ns, N])

        # Plot the predictive distributions
        if individually:
            rows = int(np.ceil(N / cols))
            for i in range(N):
                plt.subplot(rows, cols, i + 1)
                plot_dist(samples[:, i], **kwargs)
                plt.xlabel("Predicted dependent variable value for " + str(i))
            plt.tight_layout()
        else:
            plot_dist(samples, **kwargs)
            plt.xlabel("Predicted dependent variable value")

    def _get_y(self, x, y):
        """Get y, even when x is a DataGenerator and y is None"""
        if y is not None:
            return y
        else:
            y_true = [d for _, d in make_generator(x, y, test=True)]
            return np.concatenate(to_numpy(y_true), axis=0)

    def predictive_prc(self, x, y=None, n=1000, batch_size=None):
        r"""Compute the percentile of each observation along the posterior
        predictive distribution.

        TODO: Docs...  Returns a percentile between 0 and 1

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 1000
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        prcs : |ndarray| of float between 0 and 1
        """

        # Need both x and y data
        if y is None and not isinstance(x, DataGenerator):
            raise TypeError("need both x and y to compute predictive prc")

        # Compute in batches?
        if batch_size is not None:
            return np.concatenate(
                [
                    self.predictive_prc(x_data, y_data, n=n)
                    for x_data, y_data in make_generator(
                        x, y, batch_size=batch_size
                    )
                ],
                axis=0,
            )

        # Sample from the predictive distribution
        samples = self.predictive_sample(x, n=n, batch_size=batch_size)

        # Independent variable must be scalar
        if samples.ndim > 2 and any(e > 1 for e in samples.shape[2:]):
            raise NotImplementedError(
                "only scalar dependent variables are supported"
            )

        # Reshape
        Ns = samples.shape[0]
        N = samples.shape[1]
        samples = samples.reshape([Ns, N])
        y = self._get_y(x, y).reshape([1, N])

        # Percentiles of true y data along predictive distribution
        prcs = np.argmax(np.sort(samples, axis=0) > y, axis=0) / Ns

        # Argmax returns 0 when all samples are less than true value!
        prcs[np.reshape(np.max(samples, axis=0) < y, [N])] = 1.0

        # Return percentiles
        return prcs.reshape([N, 1])

    def pred_dist_covered(
        self, x, y=None, n: int = 1000, ci: float = 0.95, batch_size=None
    ):
        r"""Compute whether each observation was covered by a given confidence
        interval.

        TODO: Docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 1000
        ci : float between 0 and 1
            Confidence interval to use.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        TODO
        """

        # Check values
        if n < 1:
            raise ValueError("n must be greater than 0")
        if ci < 0.0 or ci > 1.0:
            raise ValueError("ci must be between 0 and 1")

        # Compute the predictive percentile of each observation
        pred_prcs = self.predictive_prc(x, y=y, n=n, batch_size=batch_size)

        # Determine what samples fall in the inner ci proportion
        lb = (1.0 - ci) / 2.0
        ub = 1.0 - lb
        return (pred_prcs >= lb) & (pred_prcs < ub)

    def pred_dist_coverage(self, x, y=None, n=1000, ci=0.95, batch_size=None):
        r"""Compute what percent of samples are covered by a given confidence
        interval.

        TODO: Docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 1000
        ci : float between 0 and 1
            Confidence interval to use.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        prc_covered : float between 0 and 1
            Proportion of the samples which were covered by the predictive
            distribution's confidence interval.
        """
        return self.pred_dist_covered(
            x, y=y, n=n, ci=ci, batch_size=batch_size
        ).mean()

    def coverage_by(
        self,
        x_by,
        x,
        y=None,
        n: int = 1000,
        ci: float = 0.95,
        bins: int = 30,
        plot: bool = True,
        ideal_line_kwargs: dict = {},
        batch_size=None,
        **kwargs,
    ):
        r"""Compute and plot the coverage of a given confidence interval
        of the posterior predictive distribution as a function of specified
        independent variables.

        TODO: Docs...

        Parameters
        ----------
        x_by : int or str or list of int or list of str
            Which independent variable(s) to plot the log probability as a
            function of.  That is, which columns in ``x`` to plot by.
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        ci : float between 0 and 1
            Inner percentile to find the coverage of.  For example, if
            ``ci=0.95``, will compute the coverage of the inner 95% of the
            posterior predictive distribution.
        bins : int
            Number of bins to use for x_by
        ideal_line_kwargs : dict
            Dict of args to pass to matplotlib.pyplot.plot for ideal coverage
            line.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).
        **kwargs
            Additional keyword arguments are passed to plot_by

        Returns
        -------
        xo : |ndarray|
            Values of x_by corresponding to bin centers.
        co : |ndarray|
            Coverage of the ``ci`` confidence interval of the predictive
            distribution in each bin.
        """

        # Compute whether each sample was covered by the predictive interval
        covered = self.pred_dist_covered(
            x, y=y, n=n, ci=ci, batch_size=batch_size
        )

        # Plot coverage proportion as a fn of x_by cols of x
        xo, co = plot_by(x_by, 100 * covered, label="Actual", **kwargs)

        # Line kwargs
        if "linestyle" not in ideal_line_kwargs:
            ideal_line_kwargs["linestyle"] = "--"
        if "color" not in ideal_line_kwargs:
            ideal_line_kwargs["color"] = "k"

        # Also plot ideal line
        plt.axhline(100 * ci, label="Ideal", **ideal_line_kwargs)
        plt.legend()
        plt.ylabel(str(100 * ci) + "% predictive interval coverage")
        plt.xlabel("Independent variable")

        return xo, co

    def r_squared(self, x, y=None, n=1000, batch_size=None):
        r"""Compute the Bayesian R-squared distribution (Gelman et al., 2018).

        TODO: more info


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        n : int
            Number of posterior draws to use for computing the r-squared
            distribution.  Default = `1000`.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        |ndarray|
            Samples from the r-squared distribution.  Size: ``(num_samples,)``.


        Examples
        --------
        TODO: Docs...


        References
        ----------

        - Andrew Gelman, Ben Goodrich, Jonah Gabry, & Aki Vehtari.
          `R-squared for Bayesian regression models. <https://doi.org/10.1080/00031305.2018.1549100>`_
          *The American Statistician*, 2018.

        """

        # Get true y values
        y_true = self._get_y(x, y)

        # Predict y with samples from the posterior distribution
        y_pred = self.epistemic_sample(x, n=n, batch_size=batch_size)

        # Compute Bayesian R^2
        v_fit = np.var(y_pred, axis=1)
        v_res = np.var(y_pred - np.expand_dims(y_true, 0), axis=1)
        return v_fit / (v_fit + v_res)

    def r_squared_plot(
        self, x, y=None, n=1000, style="hist", batch_size=None, **kwargs
    ):
        r"""Plot the Bayesian R-squared distribution.

        See :meth:`~r_squared` for more info on the Bayesian R-squared metric.

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        n : int
            Number of posterior draws to use for computing the r-squared
            distribution.  Default = `1000`.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).
        **kwargs
            Additional keyword arguments are passed to :func:`.plot_dist`

        Example
        -------

        TODO

        """
        r2 = self.r_squared(x, y, n=n, batch_size=batch_size)
        plot_dist(r2, style=style, **kwargs)
        plt.xlabel("Bayesian R squared")

    def residuals(self, x, y=None, batch_size=None):
        r"""Compute the residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        |ndarray|
            The residuals.

        Example
        -------

        TODO

        """
        y_true = self._get_y(x, y)
        y_pred = self.predict(x, batch_size=batch_size)
        return y_true - y_pred

    def residuals_plot(self, x, y=None, batch_size=None, **kwargs):
        r"""Plot the distribution of residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).
        **kwargs
            Additional keyword arguments are passed to :func:`.plot_dist`

        Example
        -------

        TODO

        """
        res = self.residuals(x, y, batch_size=batch_size)
        plot_dist(res, **kwargs)
        plt.xlabel("Residual (True - Predicted)")

    def calibration_curve(self, x, y, n=1000, resolution=100, batch_size=None):
        r"""Compute the regression calibration curve (Kuleshov et al., 2018).

        The regression calibration curve compares the empirical cumulative
        probability to the cumulative probability predicted by a regression
        model (Kuleshov et al., 2018).  First, a vector :math:`p` of :math:`m`
        confidence levels are chosen, which correspond to the predicted
        cumulative probabilities:

        .. math::

            0 \leq p_1 \leq p_2 \leq \ldots \leq p_m \leq 1

        Then, a vector of empirical frequencies :math:`\hat{p}` at each of the
        predicted frequencies is computed by using validation data:

        .. math::

            \hat{p}_j = \frac{1}{N} \sum_{i=1}^N [ P_M(x_i \leq y_i) \leq p_j ]

        where :math:`N` is the number of validation datapoints, :math:`P_M(x_i
        \leq y_i)` is the model's predicted cumulative probability of datapoint
        :math:`i` (i.e., the percentile along the model's predicted probability
        distribution at which the true value of :math:`y_i` falls), and
        :math:`\sum_i [ a_i \leq b_i ]` is just the count of elements of
        :math:`a` which are less than corresponding elements in :math:`b`.

        The calibration curve then plots :math:`p` against :math:`\hat{p}`.


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        n : int
            Number of samples to draw from the model for computing the
            predictive percentile.  Default = 1000
        resolution : int
            Number of confidence levels to evaluate at.  This corresponds to
            the :math:`m` parameter in section 3.5 of (Kuleshov et al., 2018).
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        p : |ndarray|
            The predicted cumulative frequencies, :math:`p`.
        p_hat : |ndarray|
            The empirical cumulative frequencies, :math:`\hat{p}`.

        Example
        -------

        Supposing we have some training data (``x_train`` and ``y_train``) and
        validation data (``x_val`` and ``y_val``), and have already fit a model
        to the training data,

        .. code-block:: python3

            model = # some ProbFlow model...
            model.fit(x_train, y_train)

        Then we can compute the calibration curve with
        :meth:`~calibration_curve`:

        .. code-block:: python3

            p_pred, p_empirical = model.calibration_curve(x_val, y_val)

        The returned values can be used directly or plotted against one another
        to get the calibration curve (as in Figure 3 in  Kuleshov et al., 2018)

        .. code-block:: python3

            import matplotlib.pyplot as plt
            plt.plot(p_pred, p_empirical)

        Or, even more simply, just use :meth:`~calibration_curve_plot`.


        See also
        --------

        * :meth:`~calibration_curve_plot`
        * :meth:`~expected_calibration_error`


        References
        ----------

        - Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon.
          `Accurate Uncertainties for Deep Learning Using Calibrated Regression
          <https://arxiv.org/abs/1807.00263>`_, 2018.

        """
        pred_prc = self.predictive_prc(x, y, n=n, batch_size=batch_size)
        p = np.linspace(0, 1, resolution + 2)[1:-1]
        p_hat = np.array([np.mean(pred_prc < tp) for tp in p])
        return p, p_hat

    def calibration_curve_plot(
        self, x, y, n=1000, resolution=100, batch_size=None, **kwargs
    ):
        r"""Plot the regression calibration curve.

        See :meth:`~calibration_curve` for more info about the regression
        calibration curve.

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        n : int
            Number of samples to draw from the model for computing the
            predictive percentile.  Default = 1000
        resolution : int
            Number of confidence levels to evaluate at.  This corresponds to
            the :math:`m` parameter in section 3.5 of (Kuleshov et al., 2018).
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).
        **kwargs
            Additional keyword arguments are passed to :func:`.plot_dist`


        See also
        --------

        * :meth:`~calibration_curve`
        * :meth:`~expected_calibration_error`

        """
        p, p_hat = self.calibration_curve(
            x, y, n=n, resolution=resolution, batch_size=batch_size
        )
        plt.plot(p, p_hat, **kwargs)
        plt.xlabel("Predicted cumulative probability")
        plt.ylabel("Empirical cumulative probability")

    def _calibration_metric(self, metric: str, p, p_hat):
        if metric == "msce":
            return np.mean(np.square(p - p_hat))
        elif metric == "rmsce":
            return np.sqrt(np.mean(np.square(p - p_hat)))
        elif metric == "mace":
            return np.mean(np.abs(p - p_hat))
        elif metric == "ma":
            p0 = np.concatenate([[0.0], p, [1.0]])
            p0_hat = np.concatenate([[0.0], p_hat, [1.0]])
            return np.trapz(np.abs(p0 - p0_hat), p0)
        else:
            raise ValueError(f"Unknown calibration metric {metric}")

    def calibration_metric(
        self, metric, x, y=None, n=1000, resolution=100, batch_size=None
    ):
        r"""Compute one or more of several calibration metrics

        Regression calibration metrics measure the error between a model's
        regression calibration curve and the ideal calibration curve - i.e.,
        what the curve would be if the model were perfectly calibrated (see
        `Kuleshov et al., 2018 <https://arxiv.org/abs/1807.00263>`_ and `Chung
        et al., 2020 <https://arxiv.org/abs/2011.09588>`_).  First, a vector
        :math:`p` of :math:`m` confidence levels are chosen, which correspond
        to the predicted cumulative probabilities:

        .. math::

            0 \leq p_1 \leq p_2 \leq \ldots \leq p_m \leq 1

        Then, a vector of empirical frequencies :math:`\hat{p}` at each of the
        predicted frequencies is computed by using validation data:

        .. math::

            \hat{p}_j =
                \frac{1}{N} \sum_{i=1}^N [ P_M(x_i \leq y_i) \leq p_j ]

        where :math:`N` is the number of validation datapoints, :math:`P_M(x_i
        \leq y_i)` is the model's predicted cumulative probability of datapoint
        :math:`i` (i.e., the percentile along the model's predicted probability
        distribution at which the true value of :math:`y_i` falls), and
        :math:`\sum_i [ a_i \leq b_i ]` is just the count of elements of
        :math:`a` which are less than corresponding elements in :math:`b`.

        Various metrics can be computed from these curves to measure how
        accurately the regression model captures uncertainty:

        The **mean squared calibration error (MSCE)** is the mean squared error
        between the empirical and predicted frequencies,

        .. math::

            MSCE = \frac{1}{m} \sum_{j=1}^m (p_j - \hat{p}_j)^2

        The **root mean squared calibration error (RMSCE)** is just the square
        root of the MSCE:

        .. math::

            RMSCE = \sqrt{\frac{1}{m} \sum_{j=1}^m (p_j - \hat{p}_j)^2}

        The **mean absolute calibration error (MACE)** is the mean of the
        absolute differences between the empirical and predicted frequencies:

        .. math::

            MACE = \frac{1}{m} \sum_{j=1}^m | p_j - \hat{p}_j |

        And the **miscalibration area (MA)** is the area between the
        calibration curve and the ideal calibration curve (the identity line
        from (0, 0) to (1, 1):

        .. math::

            MA = \int_0^1 p_x - \hat{p}_x dx

        Note that MA is equal to MACE as the number of bins (set by the
        ``resolution`` keyword argument) goes to infinity.

        To choose which metric to compute, pass the name of the metric
        (``msce``, ``rmsce``, ``mace``, or ``ma``) as the first argument to
        this function (or a list of them to compute multiple).

        See `Kuleshov et al., 2018 <https://arxiv.org/abs/1807.00263>`_, `Chung
        et al., 2020 <https://arxiv.org/abs/2011.09588>`_ and the user guide
        page on :doc:`/user_guide/evaluating` for discussions of evaluating
        uncertainty estimates using calibration metrics, among other metrics.
        Note that calibration is generally less important than accuracy, but
        more important than other metrics like :meth:`~sharpness` and any
        :meth:`dispersion_metric`.


        Parameters
        ----------
        metric : str {'msce', 'rmsce', 'mace', or 'ma'} or List[str]
            Which metric(s) to compute (see above for the definition of each
            metric).  To compute multiple metrics, pass a list of the metric
            names you'd like to compute.  Available metrics are:

            * ``msce``: mean squared calibration error
            * ``rmsce``: root mean squared calibration error
            * ``mace``: mean absolute calibration error
            * ``ma``: miscalibration area

        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        n : int
            Number of samples to draw from the model for computing the
            predictive percentile.  Default = 1000
        resolution : int
            Number of confidence levels to evaluate at.  This corresponds to
            the :math:`m` parameter in section 3.5 of (Kuleshov et al., 2018).
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        float or Dict[str, float]
            The requested calibration metric.  If a list of metric names was
            passed, will return a dict whose keys are the metrics, and whose
            values are the corresponding metric values.


        Example
        -------

        Supposing we have some training data (``x_train`` and ``y_train``) and
        validation data (``x_val`` and ``y_val``), and have already fit a model
        to the training data,

        .. code-block:: python3

            model = # some ProbFlow model...
            model.fit(x_train, y_train)

        Then we can compute different calibration metrics using
        :meth:`~expected_calibration_error`.  For example, to compute the mean
        squared calibration error (MSCE):

        .. code-block:: pycon

            >>> model.calibration_metric("msce", x_val, y_val)
            0.123

        Or, to compute the mean absolute calibration error (MACE):

        .. code-block:: pycon

            >>> model.calibration_metric("mace", x_val, y_val)
            0.211

        To compute multiple metrics at the same time, pass a list of metric
        names:

        .. code-block:: pycon

            >>> model.calibration_metric(["msce", "mace"], x_val, y_val)
            {"msce": 0.123, "mace": 0.211}


        See also
        --------

        * :meth:`~calibration_curve`
        * :meth:`~calibration_curve_plot`
        * :meth:`~sharpness`
        * :meth:`~dispersion_metric`


        References
        ----------

        - Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon.
          `Accurate Uncertainties for Deep Learning Using Calibrated Regression
          <https://arxiv.org/abs/1807.00263>`_, 2018.
        - Youngseog Chung, Willie Neiswanger, Ian Char, Jeff Schneider.
          `Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty
          Quantification <https://arxiv.org/abs/2011.09588>`_, 2020.

        """
        p, p_hat = self.calibration_curve(
            x, y, n=n, resolution=resolution, batch_size=batch_size
        )
        if isinstance(metric, list):
            return {m: self._calibration_metric(m, p, p_hat) for m in metric}
        else:
            return self._calibration_metric(metric, p, p_hat)

    def sharpness(self, x, n=1000, batch_size=None):
        r"""Compute the sharpness of the model's uncertainty estimates

        The "sharpness" of a model's uncertainty estimates is the root mean of
        the estimated variances:

        .. math::

            SHA = \sqrt{\frac{1}{N} \sum_{i=1}^N \text{Var}(\hat{Y}_i)}

        See `Tran et al., 2020 <https://arxiv.org/abs/1912.10066>`_  and the
        user guide page on :doc:`/user_guide/evaluating` for discussions of
        evaluating uncertainty estimates using sharpness, among other metrics.
        Note that the sharpness should generally be one of the later things you
        consider - accuracy and calibration usually being more important.


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        n : int
            Number of samples to draw from the model.  Default = 1000
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        float
            The sharpness of the model's uncertainty estimates


        Example
        -------

        Supposing we have some training data (``x_train`` and ``y_train``) and
        validation data (``x_val`` and ``y_val``), and have already fit a model
        to the training data,

        .. code-block:: python3

            model = # some ProbFlow model...
            model.fit(x_train, y_train)

        Then we can compute the sharpness of our model's predictions with:

        .. code-block:: pycon

            >>> model.sharpness(x_val)
            0.173


        See also
        --------

        * :meth:`~calibration_metric`
        * :meth:`~dispersion_metric`


        References
        ----------

        - Kevin Tran, Willie Neiswanger, Junwoong Yoon, Qingyang Zhang, Eric
          Xing, Zachary W. Ulissi.  `Methods for comparing uncertainty
          quantifications for material property predictions
          <https://arxiv.org/abs/1912.10066>`_, 2020.

        """
        samples = self.predictive_sample(x, n=n, batch_size=batch_size)
        return np.sqrt(np.mean(np.var(samples, axis=0)))

    def _dispersion_metric(self, metric, samples):
        stds = np.std(samples, axis=0)
        if metric in ["cv", "cov", "coefficient_of_variation"]:
            return np.std(stds) / np.mean(stds)
        elif metric in ["qcd", "qcod", "quartile_coefficient_of_dispersion"]:
            q1 = np.percentile(stds, 25)
            q3 = np.percentile(stds, 75)
            return (q3 - q1) / (q3 + q1)
        else:
            raise ValueError(f"Unknown dispersion metric {metric}")

    def dispersion_metric(self, metric, x, n=1000, batch_size=None):
        r"""Compute one or more of several calibration metrics

        Dispersion metrics measure how much a model's uncertainty estimates
        vary.  There are several different dispersion metrics:

        The **coefficient of variation** (:math:`C_v`) is the ratio of the
        standard deviation to the mean (of the model's uncertainty standard
        deviations):

        .. math::

            C_v =

        The **quartile coefficient of dispersion** (:math:`QCD`) is less
        sensitive to outliers, as it simply measures the difference between the
        first and third quartile (of the model's uncertainty standard
        deviations) to their sum:

        .. math::

            QCD = \frac{Q_3 - Q_1}{Q_3 + Q_1}

        See `Tran et al., 2020 <https://arxiv.org/abs/1912.10066>`_  and the
        user guide page on :doc:`/user_guide/evaluating` for discussions of
        evaluating uncertainty estimates using dispersion metrics, among other
        metrics.  Note that dispersion metrics should generally be one of the
        last things you consider - accuracy, calibration, and sharpness usually
        being more important.


        Parameters
        ----------
        metric : str {'cv' or 'qcd'} or List[str]
            Dispersion metric to compute.  Or,
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        n : int
            Number of samples to draw from the model.  Default = 1000
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        float or Dict[str, float]
            The requested dispersion metric.  If a list of metric names was
            passed, will return a dict whose keys are the metrics, and whose
            values are the corresponding metric values.


        Example
        -------

        Supposing we have some training data (``x_train`` and ``y_train``) and
        validation data (``x_val`` and ``y_val``), and have already fit a model
        to the training data,

        .. code-block:: python3

            model = # some ProbFlow model...
            model.fit(x_train, y_train)

        Then we can compute the coefficient of variation of our model's
        predictions with:

        .. code-block:: pycon

            >>> model.dispersion_metric('cv', x_val)
            0.732

        Or the quartile coefficient of dispersion with:

        .. code-block:: pycon

            >>> model.dispersion_metric('qcd', x_val)
            0.625


        See also
        --------

        * :meth:`~calibration_metric`
        * :meth:`~sharpness`


        References
        ----------

        - Kevin Tran, Willie Neiswanger, Junwoong Yoon, Qingyang Zhang, Eric
          Xing, Zachary W. Ulissi.  `Methods for comparing uncertainty
          quantifications for material property predictions
          <https://arxiv.org/abs/1912.10066>`_, 2020.

        """
        samples = self.predictive_sample(x, n=n, batch_size=batch_size)
        if isinstance(metric, list):
            return {m: self._dispersion_metric(m, samples) for m in metric}
        else:
            return self._dispersion_metric(metric, samples)
