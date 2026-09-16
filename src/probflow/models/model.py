"""Abstract base class for probflow models."""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import probflow.utils.ops as O
from probflow.data import make_generator
from probflow.data.data_generator import DataGenerator
from probflow.modules import Module
from probflow.utils.base import BaseCallback, BaseModel
from probflow.utils.casting import to_numpy
from probflow.utils.metrics import get_metric_fn
from probflow.utils.settings import ProbflowBackend, Sampling, get_backend
from probflow.utils.shape import get_shape
from probflow.utils.training_helpers import (
    get_default_optimizer,
    get_training_step_function,
)
from probflow.utils.typing import ScalarLike, TensorLike


class Model(BaseModel, Module):
    """Abstract base class for probflow models.

    TODO

    Methods
    -------
    This class inherits several methods and properties from :class:`.Module`:

    * :attr:`~parameters`
    * :attr:`~modules`
    * :attr:`~trainable_variables`
    * :attr:`~n_parameters`
    * :attr:`~n_variables`
    * :meth:`~bayesian_update`
    * :meth:`~kl_loss`
    * :meth:`~kl_loss_batch`
    * :meth:`~reset_kl_loss`
    * :meth:`~add_kl_loss`
    * :meth:`~dumps`
    * :meth:`~save`

    and adds model-specific methods:

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
    * :meth:`~log_prob_by`
    * :meth:`~prob`
    * :meth:`~prob_by`
    * :meth:`~save`
    * :meth:`~summary`

    Users implementing child classes should implement the following methods:

    * :meth:`~__init__`
    * :meth:`~__call__`


    Example
    -------

    See the user guide section on :doc:`/user_guide/models`.

    """

    # Parameters
    _optimizer: Any = None
    _is_training: bool = False
    _learning_rate: Any = None
    _kl_weight: float = 1.0
    _current_elbo: ScalarLike = 0.0
    _train_fn: Callable
    _data: Any = None

    def log_likelihood(
        self, x_data: TensorLike | None, y_data: TensorLike | None
    ) -> ScalarLike:
        """Compute the sum log likelihood of the model given a batch of data."""
        if x_data is None and y_data is None:
            raise ValueError("x_data and y_data cannot both be None")
        elif x_data is None:
            log_likelihoods = self().log_prob(y_data)
        else:
            log_likelihoods = self(x_data).log_prob(y_data)
        return O.sum(log_likelihoods, axis=None)

    def elbo_loss(
        self,
        x_data: TensorLike | None,
        y_data: TensorLike,
        n: int,
        n_mc: int,
    ) -> ScalarLike:
        """Compute the negative ELBO, scaled to a single sample.

        Parameters
        ----------
        x_data
            The independent variable values (or None if this is a generative
            model)
        y_data
            The dependent variable values
        n : int
            Total number of datapoints in the dataset
        n_mc : int
            Number of MC samples we're taking from the posteriors
        """
        nb = get_shape(y_data)[0]  # number of samples in this batch
        if n_mc > 1:  # first dim is num MC samples if n_mc > 1
            x_data = None if x_data is None else O.expand_dims(x_data, 0)
            y_data = O.expand_dims(y_data, 0)
        log_loss = self.log_likelihood(x_data, y_data) / nb / n_mc
        kl_loss = self.kl_loss() / n + self.kl_loss_batch() / nb
        return self._kl_weight * kl_loss - log_loss

    def get_elbo(self) -> ScalarLike:
        """Get the current ELBO on training data."""
        return self._current_elbo

    def train_step(self, x_data: TensorLike, y_data: TensorLike) -> None:
        """Perform one training step."""
        elbo = self._train_fn(x_data, y_data)
        if get_backend() == ProbflowBackend.PYTORCH:
            self._current_elbo += elbo.detach().numpy()
        else:
            self._current_elbo += elbo.numpy()

    def fit(
        self,
        x: TensorLike | DataGenerator | None = None,
        y: TensorLike | None = None,
        batch_size: int = 128,
        epochs: int = 200,
        shuffle: bool = False,
        optimizer: Any = None,
        optimizer_kwargs: dict = {},
        lr: float | None = None,
        flipout: bool = True,
        num_workers: int | None = None,
        callbacks: list[BaseCallback] = [],
        eager: bool = False,
        n_mc: int = 1,
    ) -> None:
        r"""Fit the model to data.

        TODO


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values (or, if fitting a generative model,
            the dependent variable values).  Should be of shape (Nsamples,...)
        y : |None| or |ndarray| or |DataFrame| or |Series|
            Dependent variable values (or, if fitting a generative model,
            ``None``). Should be of shape (Nsamples,...).  Default = ``None``
        batch_size : int
            Number of samples to use per minibatch.
            Default = ``128``
        epochs : int
            Number of epochs to train the model.
            Default = ``200``
        shuffle : bool
            Whether to shuffle the data each epoch.  Note that this is ignored
            if ``x`` is a |DataGenerator|
            Default = ``False``
        optimizer : |None| or a backend-specific optimizer
            What optimizer to use for optimizing the variational posterior
            distributions' variables.  The default optimizer is Adam (when the
            backend is |TensorFlow| we use ``tf.keras.optimizers.Adam`` and when
            the backend is |PyTorch| the default is to use ``torch.optim.Adam`` ).
        optimizer_kwargs : dict
            Keyword arguments to pass to the optimizer.
            Default is an empty dict.
        lr : float
            Learning rate for the optimizer.
            Note that the learning rate can be updated during training using
            the set_learning_rate method.
            Default is :math:`\exp (- \log_{10} (N_p N_b))`, where :math:`N_p`
            is the number of parameters in the model, and :math:`N_b` is the
            number of samples per batch (``batch_size``).
        flipout : bool
            Whether to use flipout during training where possible
            Default = True
        num_workers : None or int > 0
            Number of parallel processes to run for loading the data.  If
            ``None``, will not use parallel processes.  If an integer, will use
            a process pool with that many processes.  Note that this parameter
            is ignored if a |DataGenerator| is passed as ``x``.  Default = None
        callbacks : List[BaseCallback]
            List of callbacks to run while training the model.  Default is
            ``[]``, i.e. no callbacks.
        eager : bool
            Whether to use eager execution.  If False, will use ``tf.function``
            (for TensorFlow) or tracing (for PyTorch) to optimize the model
            fitting.  Note that even if eager=True, you can still use eager
            execution when using the model after it is fit.  Default = False
        n_mc : int
            Number of monte carlo samples to take from the variational
            posteriors per minibatch.  The default is to just take one per
            batch.  Using a smaller number of MC samples is faster, but using a
            greater number of MC samples will decrease the variance of the
            gradients, leading to more stable parameter optimization.


        Example
        -------

        See the user guide section on :doc:`/user_guide/fitting`.
        """
        # Determine a somewhat reasonable learning rate if none was passed
        if lr is not None:
            self._learning_rate = lr
        elif self._learning_rate is None:
            default_lr = np.exp(-np.log10(self.n_parameters * batch_size))
            self._learning_rate = default_lr

        # Create DataGenerator from input data if not already
        self._data = make_generator(
            x,
            y,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        # Use default optimizer if none specified
        if optimizer is None and self._optimizer is None:
            self._optimizer = get_default_optimizer(
                self.trainable_variables,
                self._learning_rate,
                **optimizer_kwargs,
            )

        # Use eager if input type is dataframe or series
        eager_types = (pd.DataFrame, pd.Series)
        if any(isinstance(e, eager_types) for e in self._data.get_batch(0)):
            eager = True

        # Create a function to perform one training step
        self._train_fn = get_training_step_function(
            model=self,
            n=self._data.n_samples,
            flipout=flipout,
            eager=eager,
            n_mc=n_mc,
        )

        # Assign model param to callbacks
        for c in callbacks:
            c.model = self

        # Run callbacks at start of training
        self._is_training = True
        for c in callbacks:
            c.on_train_start()

        # Fit the model!
        for i in range(int(epochs)):
            # Stop training early?
            if not self._is_training:
                break

            # Run callbacks at start of epoch
            self._current_elbo = 0.0
            self._data.on_epoch_start()
            for c in callbacks:
                c.on_epoch_start()

            # Update gradients for each batch
            for x_data, y_data in self._data:
                self.train_step(x_data, y_data)

            # Run callbacks at end of epoch
            self._data.on_epoch_end()
            for c in callbacks:
                c.on_epoch_end()

        # Run callbacks at end of training
        self._is_training = False
        for c in callbacks:
            c.on_train_end()

    def stop_training(self) -> None:
        """Stop the training of the model."""
        self._is_training = False

    def set_learning_rate(self, lr: float) -> None:
        """Set the learning rate used by this model's optimizer."""
        if not isinstance(lr, float):
            raise TypeError("lr must be a float")
        else:
            self._learning_rate = lr
        if self._optimizer is None:
            return
        if get_backend() == ProbflowBackend.PYTORCH:
            for g in self._optimizer.param_groups:
                g["lr"] = self._learning_rate
        else:
            self._optimizer.learning_rate.assign(self._learning_rate)

    def set_kl_weight(self, w: float) -> None:
        """Set the weight of the KL term's contribution to the ELBO loss."""
        if not isinstance(w, float):
            raise TypeError("w must be a float")
        else:
            self._kl_weight = w

    def _sample(
        self,
        x: TensorLike | DataGenerator,
        func: Callable,
        ed: int | None = None,
        axis: int = 1,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Sample from the model."""
        samples = []
        for x_data, _ in make_generator(x, test=True, batch_size=batch_size):
            if x_data is None:
                samples += [func(self())]
            else:
                samples += [func(self(O.expand_dims(x_data, ed)))]
        return np.concatenate(to_numpy(samples), axis=axis)

    def predictive_sample(
        self,
        x: TensorLike | DataGenerator | None = None,
        n: int = 1000,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Draw samples from the posterior predictive distribution given x.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        n : int
            Number of samples to draw from the model per datapoint.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        |ndarray|
            Samples from the predictive distribution.  Size
            (num_samples, x.shape[0], ...)
        """
        with Sampling(n=n, flipout=False):
            return self._sample(
                x, lambda x: x.sample(), ed=0, batch_size=batch_size
            )

    def aleatoric_sample(
        self,
        x: TensorLike | DataGenerator | None = None,
        n: int = 1000,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Draw samples of the model's estimate given x, including only
        aleatoric uncertainty (uncertainty due to noise).

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        n : int
            Number of samples to draw from the model per datapoint.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        |ndarray|
            Samples from the predicted mean distribution.  Size
            (num_samples,x.shape[0],...)
        """
        return self._sample(x, lambda x: x.sample(n=n), batch_size=batch_size)

    def epistemic_sample(
        self,
        x: TensorLike | DataGenerator | None = None,
        n: int = 1000,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Draw samples of the model's estimate given x, including only
        epistemic uncertainty (uncertainty due to uncertainty as to the
        model's parameter values).

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        n : int
            Number of samples to draw from the model per datapoint.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        |ndarray|
            Samples from the predicted mean distribution.  Size
            (num_samples, x.shape[0], ...)
        """
        with Sampling(n=n, flipout=False):
            return self._sample(
                x, lambda x: x.mean(), ed=0, batch_size=batch_size
            )

    def predict(
        self,
        x: TensorLike | DataGenerator | None = None,
        method: str = "mean",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Predict dependent variable using the model.

        TODO... using maximum a posteriori param estimates etc


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        method : str
            Method to use for prediction.  If ``'mean'``, uses the mean of the
            predicted target distribution as the prediction.  If ``'mode'``,
            uses the mode of the distribution.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).


        Returns
        -------
        |ndarray|
            Predicted y-value for each sample in ``x``.  Of size
            (x.shape[0], y.shape[0], ..., y.shape[-1])


        Examples
        --------
        TODO: Docs...

        """
        if method == "mean":
            return self._sample(
                x, lambda x: x.mean(), axis=0, batch_size=batch_size
            )
        elif method == "mode":
            return self._sample(
                x, lambda x: x.mode(), axis=0, batch_size=batch_size
            )
        else:
            raise ValueError("unknown method " + str(method))

    def metric(
        self,
        metric: str | Callable,
        x: TensorLike | DataGenerator,
        y: TensorLike | None = None,
        batch_size: int | None = None,
    ) -> float:
        """Compute a metric of model performance.

        TODO: docs

        TODO: note that this doesn't work w/ generative models


        Parameters
        ----------
        metric : str or callable
            Metric to evaluate.  Available metrics:

            * 'lp': log likelihood sum
            * 'log_prob': log likelihood sum
            * 'accuracy': accuracy
            * 'acc': accuracy
            * 'mean_squared_error': mean squared error
            * 'mse': mean squared error
            * 'sum_squared_error': sum squared error
            * 'sse': sum squared error
            * 'mean_absolute_error': mean absolute error
            * 'mae': mean absolute error
            * 'r_squared': coefficient of determination
            * 'r2': coefficient of determination
            * 'recall': true positive rate
            * 'sensitivity': true positive rate
            * 'true_positive_rate': true positive rate
            * 'tpr': true positive rate
            * 'specificity': true negative rate
            * 'selectivity': true negative rate
            * 'true_negative_rate': true negative rate
            * 'tnr': true negative rate
            * 'precision': precision
            * 'f1_score': F-measure
            * 'f1': F-measure
            * callable: a function which takes (y_true, y_pred)

        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| to generate both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        TODO
        """
        # Get true values and predictions
        y_true = []
        y_pred = []
        for x_data, y_data in make_generator(
            x, y, test=True, batch_size=batch_size
        ):
            y_true += [y_data]
            y_pred += [self(x_data).mean()]
        y_true = np.concatenate(to_numpy(y_true), axis=0)
        y_pred = np.concatenate(to_numpy(y_pred), axis=0)

        # Compute metric between true values and predictions
        metric_fn = get_metric_fn(metric)
        return float(metric_fn(y_true, y_pred))

    def _param_data(
        self,
        params: str | list[str] | None,
        func: Callable,
    ) -> (
        dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]]
        | tuple[np.ndarray, np.ndarray]
        | np.ndarray
    ):
        """Get data about parameters in the model."""
        if isinstance(params, str):
            return next(func(p) for p in self.parameters if p.name == params)
        elif isinstance(params, list):
            return {
                p.name: func(p) for p in self.parameters if p.name in params
            }
        else:
            return {p.name: func(p) for p in self.parameters}

    def posterior_mean(
        self, params: str | list[str] | None = None
    ) -> dict[str, np.ndarray] | np.ndarray:
        """Get the mean of the posterior distribution(s).

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or List[str] or None
            Parameter name(s) for which to compute the means.
            Default is to get the mean for all parameters in the model.


        Returns
        -------
        dict
            Means of the parameter posterior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarrays| with the posterior means.  The |ndarrays| are the same
            size as each parameter. Or just the |ndarray| if
            ``params`` was a str.

        """
        return self._param_data(params, lambda x: x.posterior_mean())

    def posterior_sample(
        self, params: str | list[str] | None = None, n: int = 10000
    ) -> dict[str, np.ndarray] | np.ndarray:
        """Draw samples from parameter posteriors.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or List[str] or None
            Parameter name(s) to sample.
            Default is to get a sample for all parameters in the model.
        num_samples : int
            Number of samples to take from each posterior distribution.
            Default = 1000


        Returns
        -------
        dict
            Samples from the parameter posterior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarrays| with the posterior samples.  The |ndarrays| are of size
            (``num_samples``, param.shape). Or just the |ndarray| if
            ``params`` was a str.
        """
        return self._param_data(params, lambda x: x.posterior_sample(n=n))

    def posterior_ci(
        self,
        params: str | list[str] | None = None,
        ci: float = 0.95,
        n: int = 10000,
    ) -> (
        dict[str, tuple[np.ndarray, np.ndarray]]
        | tuple[np.ndarray, np.ndarray]
    ):
        """Posterior confidence intervals.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or List[str] or None
            Parameter name(s) to sample.
            Default is to get the confidence intervals for all parameters in
            the model.
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
        dict
            Confidence intervals of the parameter posterior distributions.
            A dictionary
            where the keys contain the parameter names and the values contain
            tuples.  The first element of each tuple is the lower bound, and
            the second element is the upper bound.
            Or just a single tuple if params was a str
        """
        return self._param_data(params, lambda x: x.posterior_ci(ci=ci, n=n))

    def prior_sample(
        self, params: str | list[str] | None = None, n: int = 10000
    ) -> dict[str, np.ndarray] | np.ndarray:
        """Draw samples from parameter priors.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or list[str] or None
            Parameter name(s) to sample.
            Default is to sample priors of all parameters in the model.
        n : int
            Number of samples to take from each prior distribution.
            Default = 10000


        Returns
        -------
        dict
            Samples from the parameter prior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarrays| with the prior samples.  The |ndarrays| are of size
            (``n``,param.shape).
        """
        return self._param_data(params, lambda x: x.prior_sample(n=n))

    def _param_plot(
        self,
        func: Callable,
        params: str | list[str] | None = None,
        cols: int = 1,
        tight_layout: bool = True,
        **kwargs,
    ) -> None:
        """Plot parameter data."""
        if params is None:
            param_list = self.parameters
        elif isinstance(params, str):
            param_list = [p for p in self.parameters if p.name == params]
        else:
            param_list = [p for p in self.parameters if p.name in params]
        rows = int(np.ceil(len(param_list) / cols))
        for iP in range(len(param_list)):
            plt.subplot(rows, cols, iP + 1)
            func(param_list[iP])
        if tight_layout:
            plt.tight_layout()

    def posterior_plot(
        self, params: str | list[str] | None = None, cols: int = 1, **kwargs
    ) -> None:
        """Plot posterior distributions of the model's parameters.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or list[str] or None
            List of names of parameters to plot.  Default is to plot the
            posterior of all parameters in the model.
        cols : int
            Divide the subplots into a grid with this many columns.
        kwargs
            Additional keyword arguments are passed to
            :meth:`.Parameter.posterior_plot`
        """
        self._param_plot(lambda x: x.posterior_plot(**kwargs), params, cols)

    def prior_plot(
        self, params: str | list[str] | None = None, cols: int = 1, **kwargs
    ) -> None:
        """Plot prior distributions of the model's parameters.

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or list[str] or None
            List of names of parameters to plot.  Default is to plot the
            prior of all parameters in the model.
        cols : int
            Divide the subplots into a grid with this many columns.
        kwargs
            Additional keyword arguments are passed to
            :meth:`.Parameter.prior_plot`
        """
        self._param_plot(lambda x: x.prior_plot(**kwargs), params, cols)

    def log_prob(
        self,
        x: TensorLike | DataGenerator | None = None,
        y: TensorLike | None = None,
        individually: bool = True,
        distribution: bool = False,
        n: int = 1000,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Compute the log probability of `y` given the model.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor
            Independent variable values of the dataset to evaluate (aka the
            "features").
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        individually : bool
            If ``individually`` is True, returns log probability for each
            sample individually, so return shape is ``(x.shape[0], ?)``.
            If ``individually`` is False, returns sum of all log probabilities,
            so return shape is ``(1, ?)``.
        distribution : bool
            If ``distribution`` is True, returns log probability posterior
            distribution (``n`` samples from the model),
            so return shape is ``(?, n)``.
            If ``distribution`` is False, returns log posterior probabilities
            using the maximum a posteriori estimate for each parameter,
            so the return shape is ``(?, 1)``.
        n : int
            Number of samples to draw for each distribution if
            ``distribution=True``.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        log_probs : |ndarray|
            Log probabilities. Shape is determined by ``individually``,
            ``distribution``, and ``n`` kwargs.
        """
        # Get a distribution of samples
        if distribution:
            with Sampling(n=1, flipout=False):
                probs = []
                for i in range(n):
                    t_probs = []
                    for x_data, y_data in make_generator(
                        x, y, batch_size=batch_size
                    ):
                        if x_data is None:
                            t_probs += [self().log_prob(y_data)]
                        else:
                            t_probs += [self(x_data).log_prob(y_data)]
                    probs += [np.concatenate(to_numpy(t_probs), axis=0)]
            probs = np.stack(to_numpy(probs), axis=probs[0].ndim)

        # Use MAP estimates
        else:
            probs = []
            for x_data, y_data in make_generator(x, y, batch_size=batch_size):
                if x_data is None:
                    probs += [self().log_prob(y_data)]
                else:
                    probs += [self(x_data).log_prob(y_data)]
            probs = np.concatenate(to_numpy(probs), axis=0)

        # Return log prob of each sample or sum of log probs
        if individually:
            return probs
        else:
            return np.sum(probs, axis=0)

    def prob(
        self,
        x: TensorLike | DataGenerator | None = None,
        y: TensorLike | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute the probability of ``y`` given the model.

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the
            "target").
        individually : bool
            If ``individually`` is True, returns probability for each
            sample individually, so return shape is ``(x.shape[0], ?)``.
            If ``individually`` is False, returns product of all probabilities,
            so return shape is ``(1, ?)``.
        distribution : bool
            If ``distribution`` is True, returns posterior probability
            distribution (``n`` samples from the model),
            so return shape is ``(?, n)``.
            If ``distribution`` is False, returns posterior probabilities
            using the maximum a posteriori estimate for each parameter,
            so the return shape is ``(?, 1)``.
        n : int
            Number of samples to draw for each distribution if
            ``distribution=True``.
        batch_size : None or int
            Compute using batches of this many datapoints.  Default is `None`
            (i.e., do not use batching).

        Returns
        -------
        probs : |ndarray|
            Probabilities. Shape is determined by ``individually``,
            ``distribution``, and ``n`` kwargs.
        """
        return np.exp(self.log_prob(x, y, **kwargs))

    def summary(self) -> None:
        """Show a summary of the model and its parameters.

        TODO

        TODO: though maybe this should be a method of module...
        model would have to add to it the observation dist

        """
        # TODO
