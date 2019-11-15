"""
Models are objects which take Tensor(s) as input, perform some computation on 
those Tensor(s), and output probability distributions.

TODO: more...

* :class:`.Model`
* :class:`.ContinuousModel`
* :class:`.DiscreteModel`
* :class:`.CategoricalModel`
* :func:`.save_model`
* :func:`.load_model`

----------

"""


__all__ = [
    'Model',
    'ContinuousModel',
    'DiscreteModel',
    'CategoricalModel',
    'save_model',
    'load_model',
]



import warnings
from typing import List, Union, Callable

import numpy as np
import matplotlib.pyplot as plt

from probflow.core.settings import get_backend
from probflow.core.settings import Sampling
from probflow.core.base import BaseParameter
from probflow.core.base import BaseDistribution
from probflow.core.base import BaseModule
from probflow.core.base import BaseDataGenerator
from probflow.core.base import BaseCallback
import probflow.core.ops as O
from probflow.utils.casting import to_numpy
from probflow.modules import Module
from probflow.utils.plotting import plot_dist
from probflow.utils.plotting import plot_discrete_dist
from probflow.utils.plotting import plot_categorical_dist
from probflow.utils.plotting import plot_by
from probflow.data import DataGenerator
from probflow.data import make_generator
from probflow.utils.metrics import get_metric_fn



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
        nb = y_data.shape[0] #number of samples in this batch
        log_loss = self.log_likelihood(x_data, y_data)/nb
        kl_loss = self.kl_loss()/n + self.kl_loss_batch()/nb
        return self._kl_weight*kl_loss - log_loss


    def get_elbo(self):
        """Get the current ELBO on training data"""
        return self._current_elbo


    def _train_step_tf(self, n, flipout):
        """Get the training step function for TensorFlow"""

        import tensorflow as tf

        #@tf.function
        def train_fn(x_data, y_data):
            self.reset_kl_loss()
            with Sampling(n=1, flipout=flipout):
                with tf.GradientTape() as tape:
                    elbo_loss = self.elbo_loss(x_data, y_data, n)
                self._current_elbo += elbo_loss.numpy()
                variables = self.trainable_variables
                gradients = tape.gradient(elbo_loss, variables)
                self._optimizer.apply_gradients(zip(gradients, variables))

        return train_fn


    def _train_step_pt(self, n, flipout):
        """Get the training step function for PyTorch"""

        import torch

        def train_fn(x_data, y_data):
            self.reset_kl_loss()
            with Sampling(n=1, flipout=flipout):
                self._optimizer.zero_grad()
                elbo_loss = self.elbo_loss(x_data, y_data, n)
                self._current_elbo += elbo_loss.detach().numpy()
                elbo_loss.backward()
                self._optimizer.step()

        return train_fn


    def train_step(self, x_data, y_data):
        """Perform one training step"""
        self._train_fn(x_data, y_data)


    def fit(self,
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
            callbacks: List[BaseCallback] = []):
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
            distributions' variables.  When the backend is |TensorFlow|
            the default is to use adam (``tf.keras.optimizers.Adam``).
            When the backend is |PyTorch| the default is to use TODO
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
        """

        # Determine a somewhat reasonable learning rate if none was passed
        if lr is not None:
            self._learning_rate = lr
        elif self._learning_rate is None:
            default_lr = np.exp(-np.log10(self.n_parameters*batch_size))
            self._learning_rate = default_lr

        # Create DataGenerator from input data if not already
        self._data = make_generator(x, y, batch_size=batch_size, 
                                    shuffle=shuffle, num_workers=num_workers)

        # Use default optimizer if none specified
        if optimizer is None and self._optimizer is None:
            if get_backend() == 'pytorch':
                import torch
                self._optimizer = torch.optim.Adam(
                    self.trainable_variables,
                    lr=self._learning_rate, **optimizer_kwargs)
            else:
                import tensorflow as tf
                self._optimizer = tf.keras.optimizers.Adam(
                    lambda: self._learning_rate, **optimizer_kwargs)

        # Create a function to perform one training step
        if get_backend() == 'pytorch':
            self._train_fn = self._train_step_pt(self._data.n_samples, flipout)
        else:
            self._train_fn = self._train_step_tf(self._data.n_samples, flipout)

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
            raise TypeError('lr must be a float')
        else:
            self._learning_rate = lr


    def set_kl_weight(self, w):
        """Set the weight of the KL term's contribution to the ELBO loss"""
        if not isinstance(w, float):
            raise TypeError('w must be a float')
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


    def predict(self, x=None, method='mean'):
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
        if method == 'mean':
            return self._sample(x, lambda x: x.mean(), axis=0)
        elif method == 'mode':
            return self._sample(x, lambda x: x.mode(), axis=0)
        else:
            raise ValueError('unknown method '+str(method))


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


    def _param_data(self,
                    params: Union[str, List[str], None],
                    func: Callable):
        """Get data about parameters in the model"""
        if isinstance(params, str):
            return [func(p) for p in self.parameters if p.name == params][0]
        elif isinstance(params, list):
            return {p.name: func(p) for p in self.parameters
                    if p.name in params}
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


    def _param_plot(self,
                    func: Callable,
                    params: Union[None, List[str]] = None,
                    cols: int = 1,
                    tight_layout: bool = True,
                    **kwargs):
        """Plot parameter data"""
        if params is None:
            param_list = self.parameters
        else:
            param_list = [p for p in self.parameters if p.name in params]
        rows = np.ceil(len(param_list)/cols)
        for iP in range(len(param_list)):
            plt.subplot(rows, cols, iP+1)
            func(param_list[iP])
        if tight_layout:
            plt.tight_layout()


    def posterior_plot(self,
                       params=None,
                       cols=1,
                       **kwargs):
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


    def prior_plot(self,
                   params=None,
                   cols=1,
                   **kwargs):
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


    def log_prob(self, 
                 x, 
                 y=None,
                 individually=True,
                 distribution=False,
                 n=1000):
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


    def prob(self, 
             x, 
             y=None,
             **kwargs):
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


    def save(self, filename):
        """Save a model to file.

        TODO

        Parameters
        ----------
        filename : str
            Save the model to a file with this name
        """
        save_model(self, filename)


    def summary(self):
        """Show a summary of the model and its parameters.

        TODO

        TODO: though maybe this should be a method of module...
        model would have to add to it the observation dist

        """
        pass
        # TODO



class ContinuousModel(Model):
    """Abstract base class for probflow models where the dependent variable 
    (the target) is continuous and 1-dimensional.

    TODO : why use this over just Model

    TODO: note that only supports discriminative models with scalar, 
    continuous dependent variables

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
    * :meth:`~pred_dist_plot`
    * :meth:`~predictive_prc`
    * :meth:`~pred_dist_covered`
    * :meth:`~pred_dist_coverage`
    * :meth:`~coverage_by`
    * :meth:`~r_squared`
    * :meth:`~r_squared_plot`
    * :meth:`~residuals`
    * :meth:`~residuals_plot`

    Example
    -------

    TODO

    """


    def _intervals(self, fn, x, side, ci=0.95, n=1000):
        """Compute intervals on some type of sample"""
        samples = fn(x, n=n)
        if side == 'lower':
            return np.percentile(samples, 100*(1.0-ci), axis=0)
        elif side == 'upper':
            return np.percentile(samples, 100*ci, axis=0)
        else:
            lb = 100*(1.0-ci)/2.0
            prcs = np.percentile(samples, [lb, 100.0-lb], axis=0)
            return prcs[0, ...], prcs[1, ...]


    def predictive_interval(self, 
                            x,
                            ci=0.95,
                            side='both',
                            n=1000):
        """Compute confidence intervals on the model's estimate of the target
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

        Returns
        -------
        lb : |ndarray|
            Lower bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='upper'``.
        ub : |ndarray|
            Upper bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='lower'``.
        """
        return self._intervals(self.predictive_sample, x, side, ci=ci, n=n)


    def aleatoric_interval(self, 
                           x,
                           ci=0.95,
                           side='both',
                           n=1000):
        """Compute confidence intervals on the model's estimate of the target
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

        Returns
        -------
        lb : |ndarray|
            Lower bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='upper'``.
        ub : |ndarray|
            Upper bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='lower'``.
        """
        return self._intervals(self.aleatoric_sample, x, side, ci=ci, n=n)


    def epistemic_interval(self, 
                           x,
                           ci=0.95,
                           side='both',
                           n=1000):
        """Compute confidence intervals on the model's estimate of the target
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

        Returns
        -------
        lb : |ndarray|
            Lower bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='upper'``.
        ub : |ndarray|
            Upper bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.  Doesn't return this if ``side='lower'``.
        """
        return self._intervals(self.epistemic_sample, x, side, ci=ci, n=n)


    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       cols=1,
                       individually=False,
                       **kwargs):
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
        individually : bool
            If ``True``, plot one subplot per datapoint in ``x``, otherwise
            plot all the predictive distributions on the same plot.
        **kwargs
            Additional keyword arguments are passed to :func:`.plot_dist`

        Example
        -------

        TODO

        """

        # Sample from the predictive distribution
        samples = self.predictive_sample(x, n=n)

        # Independent variable must be scalar
        Ns = samples.shape[0]
        N = samples.shape[1]
        if samples.ndim > 2 and any(e>1 for e in samples.shape[2:]):
            raise NotImplementedError('only scalar dependent variables are '
                                      'supported')
        else:
            samples = samples.reshape([Ns, N])

        # Plot the predictive distributions
        if individually:
            rows = np.ceil(N/cols)
            for i in range(N):
                plt.subplot(rows, cols, i+1)
                plot_dist(samples[:,i], **kwargs)
                plt.xlabel('Predicted dependent variable value for '+str(i))
            plt.tight_layout()
        else:
            plot_dist(samples, **kwargs)
            plt.xlabel('Predicted dependent variable value')


    def _get_y(self, x, y):
        """Get y, even when x is a DataGenerator and y is None"""
        if y is not None:
            return y
        else:
            y_true = [d for _, d in make_generator(x, y, test=True)]
            return np.concatenate(to_numpy(y_true), axis=0)


    def predictive_prc(self, x, y=None, n=1000):
        """Compute the percentile of each observation along the posterior
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

        Returns
        -------
        prcs : |ndarray| of float between 0 and 1
        """

        # Need both x and y data
        if y is None and not isinstance(x, DataGenerator):
            raise TypeError('need both x and y to compute predictive prc')

        # Sample from the predictive distribution
        samples = self.predictive_sample(x, n=n)
        
        # Independent variable must be scalar
        if samples.ndim > 2 and any(e>1 for e in samples.shape[2:]):
            raise NotImplementedError('only scalar dependent variables are '
                                      'supported')

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


    def pred_dist_covered(self, x, y=None, n: int = 1000, ci: float = 0.95):
        """Compute whether each observation was covered by a given confidence
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

        Returns
        -------
        TODO
        """

        # Check values
        if n < 1:
            raise ValueError('n must be greater than 0')
        if ci < 0.0 or ci > 1.0:
            raise ValueError('ci must be between 0 and 1')

        # Compute the predictive percentile of each observation
        pred_prcs = self.predictive_prc(x, y=y, n=n)

        # Determine what samples fall in the inner ci proportion
        lb = (1.0-ci)/2.0
        ub = 1.0-lb
        return (pred_prcs>=lb) & (pred_prcs<ub)


    def pred_dist_coverage(self, x, y=None, n=1000, ci=0.95):
        """Compute what percent of samples are covered by a given confidence
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


        Returns
        -------
        prc_covered : float between 0 and 1
            Proportion of the samples which were covered by the predictive
            distribution's confidence interval.
        """
        return self.pred_dist_covered(x, y=y, n=n, ci=ci).mean()


    def coverage_by(self, 
                    x_by,
                    x, 
                    y=None,
                    n: int = 1000,
                    ci: float = 0.95, 
                    bins: int = 30, 
                    plot: bool = True,
                    ideal_line_kwargs: dict = {},
                    **kwargs):
        """Compute and plot the coverage of a given confidence interval
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
        covered = self.pred_dist_covered(x, y=y, n=n, ci=ci)

        # Plot coverage proportion as a fn of x_by cols of x
        xo, co = plot_by(x_by, 100*covered, label='Actual', **kwargs)

        # Line kwargs
        if 'linestyle' not in ideal_line_kwargs:
            ideal_line_kwargs['linestyle'] = '--'
        if 'color' not in ideal_line_kwargs:
            ideal_line_kwargs['color'] = 'k'

        # Also plot ideal line
        plt.axhline(100*ci, label='Ideal', **ideal_line_kwargs)
        plt.legend()
        plt.ylabel(str(100*ci)+'% predictive interval coverage')
        plt.xlabel('Independent variable')

        return xo, co


    def r_squared(self,
                  x,
                  y=None,
                  n=1000):
        """Compute the Bayesian R-squared distribution (Gelman et al., 2018).

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
        y_pred = self.epistemic_sample(x, n=n)

        # Compute Bayesian R^2
        v_fit = np.var(y_pred, axis=1)
        v_res = np.var(y_pred-np.expand_dims(y_true, 0), axis=1)
        return v_fit/(v_fit+v_res)


    def r_squared_plot(self,
                       x,
                       y=None,
                       n=1000, 
                       style='hist',
                       **kwargs):
        """Plot the Bayesian R-squared distribution.

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
        **kwargs
            Additional keyword arguments are passed to :func:`.plot_dist`

        Example
        -------

        TODO

        """
        r2 = self.r_squared(x, y, n=n)
        plot_dist(r2, style=style, **kwargs)
        plt.xlabel('Bayesian R squared')


    def residuals(self, x, y=None):
        """Compute the residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        **kwargs
            Additional keyword arguments are passed to :func:`.plot_dist`

        Returns
        -------
        |ndarray|
            The residuals.

        Example
        -------

        TODO

        """
        y_true = self._get_y(x, y)
        y_pred = self.predict(x)
        return y_true - y_pred


    def residuals_plot(self, x, y=None, **kwargs):
        """Plot the distribution of residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series|
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        **kwargs
            Additional keyword arguments are passed to :func:`.plot_dist`

        Example
        -------

        TODO

        """
        res = self.residuals(x, y)
        plot_dist(res, **kwargs)
        plt.xlabel('Residual (True - Predicted)')



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

    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       cols=1,
                       **kwargs):
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
        if samples.ndim > 2 and any(e>1 for e in samples.shape[2:]):
            raise NotImplementedError('only discrete dependent variables are '
                                      'supported')
        else:
            samples = samples.reshape([Ns, N])

        # Plot the predictive distributions
        rows = np.ceil(N/cols)
        for i in range(N):
            plt.subplot(rows, cols, i+1)
            plot_discrete_dist(samples[:,i])
            plt.xlabel('Datapoint '+str(i))
        plt.tight_layout()


    def r_squared(self, *args, **kwargs):
        """Cannot compute R squared for a discrete model"""
        raise RuntimeError('Cannot compute R squared for a discrete model')


    def r_squared_plot(self, *args, **kwargs):
        """Cannot compute R squared for a discrete model"""
        raise RuntimeError('Cannot compute R squared for a discrete model')



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


    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       cols=1,
                       **kwargs):
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
        **kwargs
            Additional keyword arguments are passed to 
            :func:`.plot_discrete_dist`
        """

        # Sample from the predictive distribution
        samples = self.predictive_sample(x, n=n)

        # Independent variable must be scalar
        Ns = samples.shape[0]
        N = samples.shape[1]
        if samples.ndim > 2 and any(e>1 for e in samples.shape[2:]):
            raise NotImplementedError('only categorical dependent variables '
                                      'are supported')
        else:
            samples = samples.reshape([Ns, N])

        # Plot the predictive distributions
        rows = np.ceil(N/cols)
        for i in range(N):
            plt.subplot(rows, cols, i+1)
            plot_categorical_dist(samples[:,i])
            plt.xlabel('Datapoint '+str(i))
        plt.tight_layout()


    def calibration_curve(self,
                          x,
                          y=None,
                          split_by=None,
                          bins=10,
                          plot=True):
        """Plot and return calibration curve.

        Plots and returns the calibration curve (estimated
        probability of outcome vs the true probability of that
        outcome).

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        split_by : int
            Draw the calibration curve independently for datapoints
            with each unique value in `x[:,split_by]` (a categorical
            column).
        bins : int, list of float, or |ndarray|
            Bins used to compute the curve.  If an integer, will use
            `bins` evenly-spaced bins from 0 to 1.  If a vector,
            `bins` is the vector of bin edges.
        plot : bool
            Whether to plot the curve

        #TODO: split by continuous cols as well? Then will need to define bins or edges too

        TODO: Docs...

        """
        pass
        #TODO
    


def save_model(model, filename):
    """Save a model to file"""
    pass 
    # TODO



def load_model(filename):
    """Load a model from file"""
    pass 
    # TODO

