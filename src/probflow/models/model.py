from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import probflow.utils.ops as O
from probflow.data import make_generator
from probflow.modules import Module
from probflow.utils.base import BaseCallback
from probflow.utils.casting import to_numpy
from probflow.utils.metrics import get_metric_fn
from probflow.utils.settings import Sampling, get_backend


class Model(Module):
    """Abstract base class for probflow models.

    TODO

    This class inherits several methods and properties from :class:`.Module`:

    * :attr:`~parameters`
    * :attr:`~modules`
    * :attr:`~trainable_variables`
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

    """

    # Parameters
    _optimizer = None
    _is_training = False
    _learning_rate = None
    _kl_weight = 1.0
    _current_elbo = None

    def log_likelihood(self, x_data, y_data):
        """Compute the sum log likelihood of the model given a batch of data"""
        if x_data is None:
            log_likelihoods = self().log_prob(y_data)
        else:
            log_likelihoods = self(x_data).log_prob(y_data)
        return O.sum(log_likelihoods, axis=None)

    def elbo_loss(self, x_data, y_data, n):
        """Compute the negative ELBO, scaled to a single sample"""
        nb = y_data.shape[0]  # number of samples in this batch
        log_loss = self.log_likelihood(x_data, y_data) / nb
        kl_loss = self.kl_loss() / n + self.kl_loss_batch() / nb
        return self._kl_weight * kl_loss - log_loss

    def get_elbo(self):
        """Get the current ELBO on training data"""
        return self._current_elbo

    def _train_step_tensorflow(self, n, flipout=False, eager=False):
        """Get the training step function for TensorFlow"""

        import tensorflow as tf

        def train_fn(x_data, y_data):
            self.reset_kl_loss()
            with Sampling(n=1, flipout=flipout):
                with tf.GradientTape() as tape:
                    elbo_loss = self.elbo_loss(x_data, y_data, n)
                variables = self.trainable_variables
                gradients = tape.gradient(elbo_loss, variables)
                self._optimizer.apply_gradients(zip(gradients, variables))
            return elbo_loss

        if eager:
            return train_fn
        else:
            return tf.function(train_fn)

    def _train_step_pytorch(self, n, flipout=False, eager=False):
        """Get the training step function for PyTorch"""

        import torch

        if eager:

            def train_fn(x_data, y_data):
                self.reset_kl_loss()
                with Sampling(n=1, flipout=flipout):
                    self._optimizer.zero_grad()
                    elbo_loss = self.elbo_loss(x_data, y_data, n)
                    elbo_loss.backward()
                    self._optimizer.step()
                return elbo_loss

            return train_fn

        # Use PyTorch tracing, for which we have to build a module :roll_eyes:
        # and also a caching class for inputs of different sizes, b/c
        # last batch might have different number of datapoints :vomiting_face:
        else:

            class PyTorchModule(torch.nn.Module):
                def __init__(self, model):
                    super(PyTorchModule, self).__init__()
                    for i, p in enumerate(model.trainable_variables):
                        setattr(self, str(i), p)
                    self._probflow_model = model

                def elbo_loss(self, *args):
                    self._probflow_model.reset_kl_loss()
                    with Sampling(n=1, flipout=False):
                        if len(args) == 1:
                            elbo_loss = self._probflow_model.elbo_loss(
                                None, args[0], n
                            )
                        else:
                            elbo_loss = self._probflow_model.elbo_loss(
                                args[0], args[1], n
                            )
                    return elbo_loss

            class TraceCacher:
                """Cache traces for inputs of different sizes"""

                def __init__(self, model):
                    self.fns = {}  # map from input shapes to traced function
                    self.model = model

                def get_traced_module(self, *args):
                    shape = "_".join(str(e.shape) for e in args)
                    if shape in self.fns:
                        return self.fns[shape]
                    else:
                        m = PyTorchModule(self.model)
                        inputs = {"elbo_loss": args}
                        self.fns[shape] = torch.jit.trace_module(m, inputs)
                        return self.fns[shape]

                def __call__(self, *args):
                    self.model._optimizer.zero_grad()
                    traced_module = self.get_traced_module(*args)
                    elbo_loss = traced_module.elbo_loss(*args)
                    elbo_loss.backward()
                    self.model._optimizer.step()
                    return elbo_loss

            pytorch_trainer = TraceCacher(self)

            def train_fn(x_data, y_data):
                if x_data is None:
                    elbo_loss = pytorch_trainer(torch.tensor(y_data))
                else:
                    elbo_loss = pytorch_trainer(
                        torch.tensor(x_data),
                        torch.tensor(y_data),
                    )
                return elbo_loss

            return train_fn

    def train_step(self, x_data, y_data):
        """Perform one training step"""
        elbo = self._train_fn(x_data, y_data)
        if get_backend() == "pytorch":
            self._current_elbo += elbo.detach().numpy()
        else:
            self._current_elbo += elbo.numpy()

    def fit(
        self,
        x,
        y=None,
        batch_size: int = 128,
        epochs: int = 200,
        shuffle: bool = False,
        optimizer=None,
        optimizer_kwargs: dict = {},
        lr: float = None,
        flipout: bool = True,
        num_workers: int = None,
        callbacks: List[BaseCallback] = [],
        eager: bool = False,
    ):
        r"""Fit the model to data

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
            Default = ``True``
        optimizer : |None| or a backend-specific optimizer
            What optimizer to use for optimizing the variational posterior
            distributions' variables.  When the backend is |TensorFlow| the
            default is to use adam (``tf.keras.optimizers.Adam``).  When the
            backend is |PyTorch| the default is to use TODO
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
            ``None``, will not use parallel processes.  If an integer, will
            use a process pool with that many processes.
            Default = None
        callbacks : List[BaseCallback]
            List of callbacks to run while training the model
        eager : bool
            Whether to use eager execution.  If False, will use ``tf.function``
            (for TensorFlow) or tracing (for PyTorch) to optimize the model
            fitting.  Note that even if eager=True, you can still use eager
            execution when using the model after it is fit.  Default = False
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
            if get_backend() == "pytorch":
                import torch

                self._optimizer = torch.optim.Adam(
                    self.trainable_variables,
                    lr=self._learning_rate,
                    **optimizer_kwargs
                )
            else:
                import tensorflow as tf

                self._optimizer = tf.keras.optimizers.Adam(
                    lambda: self._learning_rate, **optimizer_kwargs
                )

        # Use eager if input type is dataframe or series
        eager_types = (pd.DataFrame, pd.Series)
        if any(isinstance(e, eager_types) for e in self._data.get_batch(0)):
            eager = True

        # Create a function to perform one training step
        if get_backend() == "pytorch":
            self._train_fn = self._train_step_pytorch(
                self._data.n_samples, flipout, eager=eager
            )
        else:
            self._train_fn = self._train_step_tensorflow(
                self._data.n_samples, flipout, eager=eager
            )

        # Assign model param to callbacks
        for c in callbacks:
            c.model = self

        # Fit the model!
        self._is_training = True
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

    def stop_training(self):
        """Stop the training of the model"""
        self._is_training = False

    def set_learning_rate(self, lr):
        """Set the learning rate used by this model's optimizer"""
        if not isinstance(lr, float):
            raise TypeError("lr must be a float")
        else:
            self._learning_rate = lr

    def set_kl_weight(self, w):
        """Set the weight of the KL term's contribution to the ELBO loss"""
        if not isinstance(w, float):
            raise TypeError("w must be a float")
        else:
            self._kl_weight = w

    def _sample(self, x, func, ed=None, axis=1):
        """Sample from the model"""
        samples = []
        for x_data, y_data in make_generator(x, test=True):
            if x_data is None:
                samples += [func(self())]
            else:
                samples += [func(self(O.expand_dims(x_data, ed)))]
        return np.concatenate(to_numpy(samples), axis=axis)

    def predictive_sample(self, x=None, n=1000):
        """Draw samples from the posterior predictive distribution given x

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        n : int
            Number of samples to draw from the model per datapoint.


        Returns
        -------
        |ndarray|
            Samples from the predictive distribution.  Size
            (num_samples, x.shape[0], ...)
        """
        with Sampling(n=n, flipout=False):
            return self._sample(x, lambda x: x.sample(), ed=0)

    def aleatoric_sample(self, x=None, n=1000):
        """Draw samples of the model's estimate given x, including only
        aleatoric uncertainty (uncertainty due to noise)

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        n : int
            Number of samples to draw from the model per datapoint.


        Returns
        -------
        |ndarray|
            Samples from the predicted mean distribution.  Size
            (num_samples,x.shape[0],...)
        """
        return self._sample(x, lambda x: x.sample(n=n))

    def epistemic_sample(self, x=None, n=1000):
        """Draw samples of the model's estimate given x, including only
        epistemic uncertainty (uncertainty due to uncertainty as to the
        model's parameter values)

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the
            "features").
        n : int
            Number of samples to draw from the model per datapoint.


        Returns
        -------
        |ndarray|
            Samples from the predicted mean distribution.  Size
            (num_samples, x.shape[0], ...)
        """
        with Sampling(n=n, flipout=False):
            return self._sample(x, lambda x: x.mean(), ed=0)

    def predict(self, x=None, method="mean"):
        """Predict dependent variable using the model

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
            return self._sample(x, lambda x: x.mean(), axis=0)
        elif method == "mode":
            return self._sample(x, lambda x: x.mode(), axis=0)
        else:
            raise ValueError("unknown method " + str(method))

    def metric(self, metric, x, y=None):
        """Compute a metric of model performance

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

        Returns
        -------
        TODO
        """

        # Get true values and predictions
        y_true = []
        y_pred = []
        for x_data, y_data in make_generator(x, y, test=True):
            y_true += [y_data]
            y_pred += [self(x_data).mean()]
        y_true = np.concatenate(to_numpy(y_true), axis=0)
        y_pred = np.concatenate(to_numpy(y_pred), axis=0)

        # Compute metric between true values and predictions
        metric_fn = get_metric_fn(metric)
        return metric_fn(y_true, y_pred)

    def _param_data(self, params: Union[str, List[str], None], func: Callable):
        """Get data about parameters in the model"""
        if isinstance(params, str):
            return [func(p) for p in self.parameters if p.name == params][0]
        elif isinstance(params, list):
            return {
                p.name: func(p) for p in self.parameters if p.name in params
            }
        else:
            return {p.name: func(p) for p in self.parameters}

    def posterior_mean(self, params=None):
        """Get the mean of the posterior distribution(s)

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

    def posterior_sample(self, params=None, n=10000):
        """Draw samples from parameter posteriors

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

    def posterior_ci(self, params=None, ci=0.95, n=10000):
        """Posterior confidence intervals

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

    def prior_sample(self, params=None, n=10000):
        """Draw samples from parameter priors

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : list
            List of parameter names to sample.  Each element should be a str.
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
        params: Union[None, List[str]] = None,
        cols: int = 1,
        tight_layout: bool = True,
        **kwargs
    ):
        """Plot parameter data"""
        if params is None:
            param_list = self.parameters
        else:
            param_list = [p for p in self.parameters if p.name in params]
        rows = np.ceil(len(param_list) / cols)
        for iP in range(len(param_list)):
            plt.subplot(rows, cols, iP + 1)
            func(param_list[iP])
        if tight_layout:
            plt.tight_layout()

    def posterior_plot(self, params=None, cols=1, **kwargs):
        """Plot posterior distributions of the model's parameters

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or list or None
            List of names of parameters to plot.  Default is to plot the
            posterior of all parameters in the model.
        cols : int
            Divide the subplots into a grid with this many columns.
        kwargs
            Additional keyword arguments are passed to
            :meth:`.Parameter.posterior_plot`
        """
        self._param_plot(lambda x: x.posterior_plot(**kwargs), params, cols)

    def prior_plot(self, params=None, cols=1, **kwargs):
        """Plot prior distributions of the model's parameters

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or list or None
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
        self, x, y=None, individually=True, distribution=False, n=1000
    ):
        """Compute the log probability of `y` given the model

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

        Returns
        -------
        log_probs : |ndarray|
            Log probabilities. Shape is determined by ``individually``,
            ``distribution``, and ``n`` kwargs.
        """

        # Get a distribution of samples
        if distribution:
            with Sampling():
                probs = []
                for i in range(n):
                    t_probs = []
                    for x_data, y_data in make_generator(x, y):
                        if x_data is None:
                            t_probs += [self().log_prob(y_data)]
                        else:
                            t_probs += [self(x_data).log_prob(y_data)]
                    probs += [np.concatenate(to_numpy(t_probs), axis=0)]
            probs = np.stack(to_numpy(probs), axis=probs[0].ndim)

        # Use MAP estimates
        else:
            probs = []
            for x_data, y_data in make_generator(x, y):
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

    def prob(self, x, y=None, **kwargs):
        """Compute the probability of ``y`` given the model

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

        Returns
        -------
        probs : |ndarray|
            Probabilities. Shape is determined by ``individually``,
            ``distribution``, and ``n`` kwargs.
        """
        return np.exp(self.log_prob(x, y, **kwargs))

    def summary(self):
        """Show a summary of the model and its parameters.

        TODO

        TODO: though maybe this should be a method of module...
        model would have to add to it the observation dist

        """
        pass
        # TODO
