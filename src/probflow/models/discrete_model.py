import matplotlib.pyplot as plt
import numpy as np

from probflow.utils.plotting import plot_discrete_dist

from .continuous_model import ContinuousModel


class DiscreteModel(ContinuousModel):
    """Abstract base class for probflow models where the dependent variable
    (the target) is discrete (e.g. drawn from a Poisson distribution).

    TODO : why use this over just Model

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

    as well as several methods from :class:`.ContinuousModel`:

    * :meth:`~predictive_interval`
    * :meth:`~predictive_prc`
    * :meth:`~pred_dist_covered`
    * :meth:`~pred_dist_coverage`
    * :meth:`~coverage_by`
    * :meth:`~residuals`

    but overrides the following discrete-model-specific methods:

    * :meth:`~pred_dist_plot`
    * :meth:`~residuals_plot`

    Note that :class:`.DiscreteModel` does *not* implement :meth:`~r_squared`
    or :meth:`~r_squared_plot`.

    Example
    -------

    TODO

    """

    def pred_dist_plot(self, x, n=10000, cols=1, **kwargs):
        """Plot posterior predictive distribution from the model given ``x``.

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
        **kwargs
            Additional keyword arguments are passed to
            :func:`.plot_discrete_dist`
        """

        # Sample from the predictive distribution
        samples = self.predictive_sample(x, n=n)

        # Independent variable must be scalar
        Ns = samples.shape[0]
        N = samples.shape[1]
        if samples.ndim > 2 and any(e > 1 for e in samples.shape[2:]):
            raise NotImplementedError(
                "only discrete dependent variables are " "supported"
            )
        else:
            samples = samples.reshape([Ns, N])

        # Plot the predictive distributions
        rows = np.ceil(N / cols)
        for i in range(N):
            plt.subplot(rows, cols, i + 1)
            plot_discrete_dist(samples[:, i])
            plt.xlabel("Datapoint " + str(i))
        plt.tight_layout()

    def r_squared(self, *args, **kwargs):
        """Cannot compute R squared for a discrete model"""
        raise RuntimeError("Cannot compute R squared for a discrete model")

    def r_squared_plot(self, *args, **kwargs):
        """Cannot compute R squared for a discrete model"""
        raise RuntimeError("Cannot compute R squared for a discrete model")
