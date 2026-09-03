import matplotlib.pyplot as plt
import numpy as np

from probflow.data.data_generator import DataGenerator
from probflow.utils.plotting import plot_categorical_dist
from probflow.utils.typing import TensorLike

from .model import Model


class CategoricalModel(Model):
    """Abstract base class for probflow models where the dependent variable
    (the target) is categorical (e.g. drawn from a Bernoulli distribution).

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

    and adds the following categorical-model-specific methods:

    * :meth:`~pred_dist_plot`
    * :meth:`~calibration_curve`

    Example
    -------

    TODO

    """

    def pred_dist_plot(self, x: TensorLike | DataGenerator, n: int = 10000, cols: int = 1, batch_size: int | None = None, **kwargs):
        """Plot posterior predictive distribution from the model given ``x``.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        n : int
            Number of samples to draw from the model given ``x``.
            Default = 10000
        cols : int
            Divide the subplots into a grid with this many columns (if
            ``individually=True``.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).
        **kwargs
            Additional keyword arguments are passed to
            :func:`.plot_categorical_dist`
        """

        # Sample from the predictive distribution
        samples = self.predictive_sample(x, n=n, batch_size=batch_size)

        # Independent variable must be scalar
        Ns = samples.shape[0]
        N = samples.shape[1]
        if samples.ndim > 2 and any(e > 1 for e in samples.shape[2:]):
            raise NotImplementedError(
                "only categorical dependent variables are supported"
            )
        else:
            samples = samples.reshape([Ns, N])

        # Plot the predictive distributions
        rows = int(np.ceil(N / cols))
        for i in range(N):
            plt.subplot(rows, cols, i + 1)
            plot_categorical_dist(samples[:, i])
            plt.xlabel("Datapoint " + str(i))
        plt.tight_layout()
