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
from typing import List

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
from probflow.modules import Module
from probflow.utils.plotting import plot_dist
from probflow.data import DataGenerator
from probflow.data import make_generator
from probflow.utils.metrics import get_metric_fn



class Model(Module):
    """Abstract base class for probflow models.


    TODO


    Attributes
    ----------
    parameters : list
        List of |Parameters| in the |Model|


    Methods
    -------
    kl_loss()
        The sum of the KL divergence between posteriors and priors for all
        parameters in the model
    fit(x, ...)
        Fit the model to data
    stop_training()
        Stop the training of the model
    set_learning_rate(lr)
        Set the learning rate used by this model's optimizer
    predictive_sample(x, n=1000)
        Draw samples from the predictive distribution given x
    aleatoric_sample(x, n=1000)
        Draw samples of the model's estimate given x, including only
        aleatoric uncertainty (uncertainty due to noise)
    epistemic_sample(x, n=1000)
        Draw samples of the model's estimate given x, including only
        epistemic uncertainty (uncertainty due to uncertainty as to the
        model's parameter values)
    predict(x)
        Predict dependent variable using the model
    metric(x, y, metric='log_prob')
        Compute a metric of model performance
    posterior_mean(params=None)
        Get the mean of the posterior distribution(s)
    posterior_sample(params=None, n=10000)
        Draw samples from parameter posteriors
    posterior_ci(params=None, ci=0.95, n=10000)
        Posterior confidence intervals
    posterior_plot(...)
        Plot posterior distributions of the model's parameters
    prior_sample(params=None, n=10000)
        Draw samples from parameter priors
    prior_plot(...)  
        Plot prior distributions of the model's parameters
    log_prob(x, y, individually=True, distribution=False, n=1000)
        Compute the log probability of ``y`` given the model
    log_prob_by(x_by, x, y, bins=30, plot=True)
        Log probability of observations ``y`` given the
        model, as a function of independent variable(s) ``x_by``
    prob(x, y, individually=True, distribution=False, n=1000)
        Compute the probability of ``y`` given the model
    prob_by(x_by, x, y, bins=30, plot=True)
        Probability of observations ``y`` given the
        model, as a function of independent variable(s) ``x_by``
    summary
        Show a summary of the model and its parameters.
    """


    # Whether the model is currently training
    _is_training = False


    # The current learning rate
    _learning_rate = None


    def log_likelihood(self, x_data, y_data):
        """Compute the sum log likelihood of the model given a batch of data"""
        if x_data is None:
            log_likelihoods = self().log_prob(y_data)
        else:
            log_likelihoods = self(x_data).log_prob(y_data)
        return O.sum(log_likelihoods, axis=None)


    def _train_step_tf(self, n, flipout):
        """Get the training step function for TensorFlow"""

        import tensorflow as tf

        #@tf.function
        def train_step(x_data, y_data):
            nb = y_data.shape[0] #number of samples in this batch
            self.reset_kl_loss()
            with Sampling(n=1, flipout=flipout):
                with tf.GradientTape() as tape:
                    log_loss = self.log_likelihood(x_data, y_data)/nb
                    kl_loss = self.kl_loss()/n + self.kl_loss_batch()/nb
                    elbo_loss = kl_loss - log_loss
                variables = self.trainable_variables
                gradients = tape.gradient(elbo_loss, variables)
                self._optimizer.apply_gradients(zip(gradients, variables))
            return elbo_loss

        return train_step


    def _train_step_pt(self, n, flipout):
        """Get the training step function for PyTorch"""
        raise NotImplementedError
        # TODO


    def train_step(self, x_data, y_data):
        """Perform one training step"""
        self._train_fn(x_data, y_data)


    def fit(self,
            x,
            y=None,
            batch_size: int = 128,
            epochs: int = 100,
            shuffle: bool = True,
            optimizer=None,
            optimizer_kwargs: dict = {},
            learning_rate: float = 1e-3,
            flipout: bool = True,
            callbacks: List[BaseCallback] = []):
        """Fit the model to data

        TODO

        Creates the following attributes of the Model
        * _optimizer
        * _is_training


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
            Default = ``100``
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
        learning_rate : float
            Learning rate for the optimizer.
            Note that the learning rate can be updated during training using
            the set_learning_rate method.
            Default = ``1e-3``
        flipout : bool
            Whether to use flipout during training where possible
            Default = True
        """

        # Create DataGenerator from input data if not already
        self._data = make_generator(x, y, batch_size=batch_size, 
                                    shuffle=shuffle)

        # Use default optimizer if none specified
        self._learning_rate = learning_rate
        if optimizer is None:
            if get_backend() == 'pytorch':
                import torch
                raise NotImplementedError
                # TODO
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
        for i in range(epochs):

            # Stop training early?
            if not self._is_training:
                break

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


    def predictive_sample(self, x=None, n=1000):
        """Draw samples from the predictive distribution given x

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor
            Independent variable values of the dataset to evaluate (aka the 
            "features"). 
        n : int
            Number of samples to draw from the model.


        Returns
        -------
        |ndarray|
            Samples from the predictive distribution.  Size
            (num_samples, x.shape[0], ...)
        """
        samples = []
        with Sampling(n=n, flipout=False):
            for x_data, y_data in make_generator(x, test=True):
                if x_data is None:
                    samples += [self().sample()]
                else:
                    samples += [self(O.expand_dims(x_data, 0)).sample()]
        return np.concatenate(samples, axis=1)


    def aleatoric_sample(self, x=None, n=1000):
        """Draw samples of the model's estimate given x, including only
        aleatoric uncertainty (uncertainty due to noise)

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor
            Independent variable values of the dataset to evaluate (aka the 
            "features"). 
        n : int
            Number of samples to draw from the model.


        Returns
        -------
        |ndarray|
            Samples from the predicted mean distribution.  Size
            (num_samples,x.shape[0],...)
        """
        samples = []
        for x_data, y_data in make_generator(x, test=True):
            if x_data is None:
                samples += [self().sample(n=n).numpy()]
            else:
                samples += [self(x_data).sample(n=n).numpy()]
        return np.concatenate(samples, axis=1)


    def epistemic_sample(self, x=None, n=1000):
        """Draw samples of the model's estimate given x, including only
        epistemic uncertainty (uncertainty due to uncertainty as to the
        model's parameter values)

        TODO: Docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or None
            Independent variable values of the dataset to evaluate (aka the 
            "features"). 
        n : int
            Number of samples to draw from the model.


        Returns
        -------
        |ndarray|
            Samples from the predicted mean distribution.  Size
            (num_samples, x.shape[0], ...)
        """
        samples = []
        with Sampling(n=n, flipout=False):
            for x_data, y_data in make_generator(x, test=True):
                if x_data is None:
                    samples += [self().mean().numpy()]
                else:
                    samples += [self(O.expand_dims(x_data, 0)).mean().numpy()]
        return np.concatenate(samples, axis=1)


    def predict(self, x=None):
        """Predict dependent variable using the model

        TODO... using maximum a posteriori param estimates etc


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor
            Independent variable values of the dataset to evaluate (aka the 
            "features").  


        Returns
        -------
        |ndarray|
            Predicted y-value for each sample in ``x``.  Of size
            (x.shape[0], y.shape[0], ..., y.shape[-1])


        Examples
        --------
        TODO: Docs...

        """
        preds = []
        for x_data, y_data in make_generator(x, test=True):
            if x_data is None:
                preds += [self().mean().numpy()]
            else:
                preds += [self(x_data).mean().numpy()]
        return np.concatenate(preds, axis=0)


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
            y_pred += [self(x_data).mean().numpy()]
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        # Compute metric between true values and predictions
        metric_fn = get_metric_fn(metric)
        return metric_fn(y_true, y_pred)


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
        if isinstance(params, str):
            return [p.posterior_mean() for p in self.parameters
                    if p.name == params][0]
        elif params is None:
            return {p.name: p.posterior_mean() for p in self.parameters}
        else:
            return {p.name: p.posterior_mean() for p in self.parameters
                    if p.name in params}


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
        if isinstance(params, str):
            return [p.posterior_sample(n=n) for p in self.parameters
                    if p.name == params][0]
        elif params is None:
            return {p.name: p.posterior_sample(n=n) for p in self.parameters}
        else:
            return {p.name: p.posterior_sample(n=n) for p in self.parameters
                    if p.name in params}


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
        if isinstance(params, str):
            return [p.posterior_ci(ci=ci, n=n) 
                    for p in self.parameters if p.name == params][0]
        elif params is None:
            return {p.name: p.posterior_ci(ci=ci, n=n) 
                    for p in self.parameters}
        else:
            return {p.name: p.posterior_ci(ci=ci, n=n) 
                    for p in self.parameters if p.name in params}


    def posterior_plot(self,
                       params=None,
                       cols=1,
                       tight_layout=True,
                       **kwargs):
        """Plot posterior distributions of the model's parameters

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : str or list
            List of parameters to plot.  Default is to plot the posterior of
            all parameters in the model.
        cols : int
            Divide the subplots into a grid with this many columns.
        kwargs
            Additional keyword arguments are passed to 
            :meth:`.Parameter.posterior_plot`
        """
        if params is None:
            param_list = self.parameters
        else:
            param_list = [p for p in self.parameters if p.name in params]
        rows = np.ceil(len(param_list)/cols)
        for iP in range(len(param_list)):
            plt.subplot(rows, cols, iP+1)
            param_list[iP].posterior_plot(**kwargs)
        if tight_layout:
            plt.tight_layout()


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
        if isinstance(params, str):
            return [p.prior_sample(n=n) 
                    for p in self.parameters if p.name == params][0]
        elif params is None:
            return {p.name: p.prior_sample(n=n) for p in self.parameters}
        else:
            return {p.name: p.prior_sample(n=n) for p in self.parameters
                    if p.name in params}


    def prior_plot(self,
                   params=None,
                   cols=1,
                   tight_layout=True,
                   **kwargs):
        """Plot prior distributions of the model's parameters

        TODO: Docs... params is a list of strings of params to plot


        Parameters
        ----------
        params : |None| or str or list of str
            List of parameters to plot.  Default is to plot the prior of
            all parameters in the model.
        n : int
            Number of samples to take from each prior distribution.
            Default = 10000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns.
        bins : int or list or |ndarray|
            Number of bins to use for the prior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram of the density
        """
        if params is None:
            param_list = self.parameters
        else:
            param_list = [p for p in self.parameters if p.name in params]
        rows = np.ceil(len(param_list)/cols)
        for iP in range(len(param_list)):
            plt.subplot(rows, cols, iP+1)
            param_list[iP].prior_plot(**kwargs)
        if tight_layout:
            plt.tight_layout()


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
                    for x_data, y_data in make_generator(x, y, test=True):
                        if x_data is None:
                            t_probs += [self().log_prob(y_data).numpy()]
                        else:
                            t_probs += [self(x_data).log_prob(y_data).numpy()]
                    probs += [np.concatenate(t_probs, axis=0)]
            probs = np.stack(probs, axis=probs[0].ndim)

        # Use MAP estimates
        else:
            probs = []
            for x_data, y_data in make_generator(x, y, test=True):
                if x_data is None:
                    probs += [self().log_prob(y_data).numpy()]
                else:
                    probs += [self(x_data).log_prob(y_data).numpy()]
            probs = np.concatenate(probs, axis=0)

        # Return log prob of each sample or sum of log probs
        if individually:
            return probs
        else:
            return np.sum(probs, axis=0)


    def log_prob_by(self, 
                    x_by,
                    x,
                    y=None,
                    bins=30,
                    plot=True):
        """Log probability of observations ``y`` given the
        model, as a function of independent variable(s) ``x_by``

        TODO: docs...

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
        bins : int
            Number of bins.
        plot : bool
            Whether to plot the data (if True), or just return the values.

        
        Returns
        -------
        log_probs : |ndarray|
            The average log probability as a function of ``x_by``.
            If x_by is an int or str, is of shape ``(bins,)``.
            If ``x_by`` is a list of length 2, ``prob_by`` is of shape
            ``(bins, bins)``.
        """
        pass
        # TODO
        # TODO: handle when x is a DataGenerator, or y=None


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


    def prob_by(self, 
                x_by,
                x,
                y=None,
                bins=30,
                plot=True):
        """Probability of observations ``y`` given the
        model, as a function of independent variable(s) ``x_by``

        TODO: docs...

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
        bins : int
            Number of bins.
        plot : bool
            Whether to plot the data (if True), or just return the values.

        
        Returns
        -------
        log_probs : |ndarray|
            The average log probability as a function of ``x_by``.
            If x_by is an int or str, is of shape ``(bins,)``.
            If ``x_by`` is a list of length 2, ``prob_by`` is of shape
            ``(bins, bins)``.
        """
        pass
        # TODO
        # TODO: handle when x is a DataGenerator, or y=None


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


    TODO


    Methods
    -------

    This class inherits several methods from :class:`.Model`:

    * :func:`~probflow.models.ContinuousModel.__init__`
    * :func:`~probflow.models.ContinuousModel.__call__`
    * :func:`~probflow.models.ContinuousModel.parameters`
    * :func:`~probflow.models.ContinuousModel.kl_loss`
    * :func:`~probflow.models.ContinuousModel.fit`
    * :func:`~probflow.models.ContinuousModel.stop_training`
    * :func:`~probflow.models.ContinuousModel.set_learning_rate`
    * :func:`~probflow.models.ContinuousModel.predictive_distribution`
    * :func:`~probflow.models.ContinuousModel.mean_distribution`
    * :func:`~probflow.models.ContinuousModel.predict`
    * :func:`~probflow.models.ContinuousModel.metric`
    * :func:`~probflow.models.ContinuousModel.posterior_mean`
    * :func:`~probflow.models.ContinuousModel.posterior_sample`
    * :func:`~probflow.models.ContinuousModel.posterior_plot`
    * :func:`~probflow.models.ContinuousModel.prior_sample`
    * :func:`~probflow.models.ContinuousModel.prior_plot`
    * :func:`~probflow.models.ContinuousModel.log_prob`
    * :func:`~probflow.models.ContinuousModel.log_prob_by`
    * :func:`~probflow.models.ContinuousModel.prob`
    * :func:`~probflow.models.ContinuousModel.prob_by`
    * :func:`~probflow.models.ContinuousModel.summary`

    and adds the following continuous-model-specific methods:

    * :func:`~probflow.models.ContinuousModel.confidence_intervals`
    * :func:`~probflow.models.ContinuousModel.pred_dist_plot`
    * :func:`~probflow.models.ContinuousModel.pred_dist_prc`
    * :func:`~probflow.models.ContinuousModel.pred_dist_covered`
    * :func:`~probflow.models.ContinuousModel.pred_dist_coverage`
    * :func:`~probflow.models.ContinuousModel.coverage_by`
    * :func:`~probflow.models.ContinuousModel.calibration_curve`
    * :func:`~probflow.models.ContinuousModel.r_squared`
    * :func:`~probflow.models.ContinuousModel.residuals`
    * :func:`~probflow.models.ContinuousModel.residuals_plot`

    """


    def confidence_intervals(self, 
                             x,
                             ci=0.95,
                             n=1000):
        """Compute confidence intervals on predictions for ``x``.

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
        n : int
            Number of samples from the posterior predictive distribution to
            take to compute the confidence intervals.
            Default = 1000

        Returns
        -------
        lb : |ndarray|
            Lower bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.
        ub : |ndarray|
            Upper bounds of the ``ci`` confidence intervals on the predictions
            for samples in ``x``.
        """

        # Sample from the predictive distribution
        pred_samples = self.predictive_sample(x, n=n)

        # Compute percentiles of the predictive distribution
        lb = 100*(1.0-ci)/2.0
        q = [lb, 100.0-lb]
        prcs = np.percentile(pred_samples, q, axis=0)
        return prcs[0, ...], prcs[1, ...]


    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       style='fill',
                       cols=1,
                       bins=20,
                       ci=0.0,
                       bw=0.075,
                       color=None,
                       alpha=0.4,
                       individually=False):
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
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns (if 
            ``individually=True``.
        bins : int or list or |ndarray|
            Number of bins to use for the posterior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram of the density
        individually : bool
            If ``True``, plot one subplot per datapoint in ``x``, otherwise
            plot all the predictive distributions on the same plot.
        """

        # Sample from the predictive distribution
        pred_samples = self.predictive_sample(x, n=n)

        # TODO: assumes y is scalar, add a check for that

        # Plot the predictive distributions
        N = pred_samples.shape[1]
        if individually:
            rows = np.ceil(N/cols)
            for i in range(N):
                plt.subplot(rows, cols, i+1)
                plot_dist(pred_samples[:,i], xlabel='Datapoint '+str(i), 
                          style=style, bins=bins, ci=ci, bw=bw, alpha=alpha, 
                          color=color)
        else:
            plot_dist(pred_samples, xlabel='Dependent Variable', style=style, 
                      bins=bins, ci=ci, bw=bw, alpha=alpha, color=color)


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

        # Sample from the predictive distribution
        pred_samples = self.predictive_sample(x, n=n)

        # TODO: assumes y is scalar, add a check for that

        # Return percentiles of true y data along predictive distribution
        #inds = np.argmax((np.sort(pred_samples, 0) >
        #                  y.reshape(1, x.shape[0], -1)),
        #                 axis=0)
        # TODO
        return inds/float(n)

        # TODO: check for when true y value is above max pred_samples val!
        # I think argmax returns 0 when that's the case, which is
        # obviously not what we want


    def pred_dist_covered(self, x, y=None, n=1000, ci=0.95):
        """Compute whether each observation was covered by a given confidence
        interval.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.pred_dist_covered` on a |Model|, you must
            first :meth:`.fit` it to some data.

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

        # Check types
        if not isinstance(ci, float):
            if isinstance(ci, int):
                ci = float(ci)
            else:
                raise TypeError('ci must be a float')
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
                    ci=0.95, 
                    bins=30, 
                    plot=True,
                    true_line_kwargs={},
                    ideal_line_kwargs={}):
        """Compute and plot the coverage of a given confidence interval
        of the posterior predictive distribution as a
        function of specified independent variables.

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
        plot : bool
            Whether to plot the coverage.  Default = True
        true_line_kwargs : dict
            Dict to pass to matplotlib.pyplot.plot for true coverage line
        ideal_line_kwargs : dict
            Dict of args to pass to matplotlib.pyplot.plot for ideal coverage
            line.


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
        # TODO: how to handle if x is data generator?
        """
        xo, co = plot_by(x[:, x_by], 100*covered, bins=bins,
                         plot=plot, label='Actual', **true_line_kwargs)
        """

        # Line kwargs
        if 'linestyle' not in ideal_line_kwargs:
            ideal_line_kwargs['linestyle'] = '--'
        if 'color' not in ideal_line_kwargs:
            ideal_line_kwargs['color'] = 'k'

        # Also plot ideal line
        if plot and isinstance(x_by, int):
            plt.axhline(100*ci, label='Ideal', **ideal_line_kwargs)
            plt.legend()
            plt.ylabel(str(100*ci)+'% predictive interval coverage')
            plt.xlabel('Value of '+str(x_by))

        return xo, co


    def calibration_curve(self,
                          x,
                          y=None,
                          split_by=None,
                          bins=10,
                          plot=True):
        """Plot and/or return calibration curve.

        Plots and returns the calibration curve (the percentile of the posterior
        predictive distribution on the x-axis, and the percent of samples which
        actually fall into that range on the y-axis).


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

        Returns
        -------
        cx : |ndarray|
            Vector of percentiles (the middle of each percentile
            bin).  Length is determined by `bins`.
        cy : |ndarray|
            Vector of percentages of samples which fell within each
            percentile bin of the posterior predictive distribution.

        See Also
        --------
        predictive_distribution : used to generate the posterior
            predictive distribution.

        Notes
        -----
        TODO: Docs...

        Examples
        --------
        TODO: Docs...

        """
        pass
        # TODO


    def r_squared(self,
                  x,
                  y=None,
                  n=1000,
                  plot=True):
        """Compute the Bayesian R-squared value.

        Compute the Bayesian R-squared distribution :ref:`[1] <ref_r_squared>`.
        TODO: more info and docs...


        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the 
            "target").
        n : int
            Number of posterior draws to use for computing the r-squared
            distribution.  Default = `1000`.
        plot : bool
            Whether to plot the r-squared distribution

        Returns
        -------
        |ndarray|
            Samples from the r-squared distribution.  Size: ``(num_samples,)``.

        Notes
        -----
        TODO: Docs...

        Examples
        --------
        TODO: Docs...

        References
        ----------
        .. _ref_r_squared:
        .. [1] Andrew Gelman, Ben Goodrich, Jonah Gabry, & Aki Vehtari.
            R-squared for Bayesian regression models.
            *The American Statistician*, 2018.
            https://doi.org/10.1080/00031305.2018.1549100
        """
        pass
        #TODO


    def residuals(self, x, y=None):
        """Compute the residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the 
            "target").

        Returns
        -------
        TODO

        """
        pass
        # TODO


    def residuals_plot(self, x, y=None):
        """Plot the distribution of residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the 
            "target").

        """
        pass
        # TODO



class DiscreteModel(ContinuousModel):
    """Abstract base class for probflow models where the dependent variable 
    (the target) is discrete (e.g. drawn from a Poisson distribution).


    TODO


    Methods
    -------

    This class inherits several methods from :class:`.Model`:

    * :func:`~probflow.models.DiscreteModel.__init__`
    * :func:`~probflow.models.DiscreteModel.__call__`
    * :func:`~probflow.models.DiscreteModel.parameters`
    * :func:`~probflow.models.DiscreteModel.kl_loss`
    * :func:`~probflow.models.DiscreteModel.fit`
    * :func:`~probflow.models.DiscreteModel.stop_training`
    * :func:`~probflow.models.DiscreteModel.set_learning_rate`
    * :func:`~probflow.models.DiscreteModel.predictive_distribution`
    * :func:`~probflow.models.DiscreteModel.mean_distribution`
    * :func:`~probflow.models.DiscreteModel.predict`
    * :func:`~probflow.models.DiscreteModel.metric`
    * :func:`~probflow.models.DiscreteModel.posterior_mean`
    * :func:`~probflow.models.DiscreteModel.posterior_sample`
    * :func:`~probflow.models.DiscreteModel.posterior_plot`
    * :func:`~probflow.models.DiscreteModel.prior_sample`
    * :func:`~probflow.models.DiscreteModel.prior_plot`
    * :func:`~probflow.models.DiscreteModel.log_prob`
    * :func:`~probflow.models.DiscreteModel.log_prob_by`
    * :func:`~probflow.models.DiscreteModel.prob`
    * :func:`~probflow.models.DiscreteModel.prob_by`
    * :func:`~probflow.models.DiscreteModel.summary`

    and also inherits several methods from :class:`.ContinuousModel`:

    * :func:`~probflow.models.DiscreteModel.confidence_intervals`
    * :func:`~probflow.models.DiscreteModel.pred_dist_prc`
    * :func:`~probflow.models.DiscreteModel.pred_dist_covered`
    * :func:`~probflow.models.DiscreteModel.pred_dist_coverage`
    * :func:`~probflow.models.DiscreteModel.coverage_by`
    * :func:`~probflow.models.DiscreteModel.calibration_curve`
    * :func:`~probflow.models.DiscreteModel.residuals`

    but overrides the following discrete-model-specific methods:

    * :func:`~probflow.models.DiscreteModel.pred_dist_plot`
    * :func:`~probflow.models.DiscreteModel.residuals_plot`

    """

    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       style='fill',
                       cols=1,
                       bins=20,
                       ci=0.0,
                       bw=0.075,
                       color=None,
                       alpha=0.4,
                       individually=False):
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
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns (if 
            ``individually=True``.
        bins : int or list or |ndarray|
            Number of bins to use for the posterior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram of the density
        individually : bool
            If ``True``, plot one subplot per datapoint in ``x``, otherwise
            plot all the predictive distributions on the same plot.
        """

        # Sample from the predictive distribution
        pred_samples = self.predictive_sample(x, n=n)

        # TODO: assumes y is scalar, add a check for that

        # TODO: plot discretely

        # Plot the predictive distributions
        N = pred_samples.shape[1]
        if individually:
            rows = np.ceil(N/cols)
            for i in range(N):
                plt.subplot(rows, cols, i+1)
                #plot_dist(pred_samples[:,i], xlabel='Datapoint '+str(i), 
                #          style=style, bins=bins, ci=ci, bw=bw, alpha=alpha, 
                #          color=color)
        else:
            #plot_dist(pred_samples, xlabel='Dependent Variable', style=style, 
            #          bins=bins, ci=ci, bw=bw, alpha=alpha, color=color)
            pass


    def r_squared(self, *args, **kwargs):
        """Cannot compute R squared for a discrete model"""
        raise RuntimeError('Cannot compute R squared for a discrete model')


    def residuals_plot(self, x, y=None):
        """Plot the distribution of residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
            Independent variable values of the dataset to evaluate (aka the 
            "features").  Or a |DataGenerator| for both x and y.
        y : |ndarray| or |DataFrame| or |Series| or Tensor
            Dependent variable values of the dataset to evaluate (aka the 
            "target").

        """
        pass
        # TODO: plot discretely



class CategoricalModel(Model):
    """Abstract base class for probflow models where the dependent variable 
    (the target) is categorical (e.g. drawn from a Bernoulli distribution).


    TODO


    Methods
    -------

    This class inherits several methods from :class:`.Model`:

    * :func:`~probflow.models.CategoricalModel.__init__`
    * :func:`~probflow.models.CategoricalModel.__call__`
    * :func:`~probflow.models.CategoricalModel.parameters`
    * :func:`~probflow.models.CategoricalModel.kl_loss`
    * :func:`~probflow.models.CategoricalModel.fit`
    * :func:`~probflow.models.CategoricalModel.stop_training`
    * :func:`~probflow.models.CategoricalModel.set_learning_rate`
    * :func:`~probflow.models.CategoricalModel.predictive_distribution`
    * :func:`~probflow.models.CategoricalModel.mean_distribution`
    * :func:`~probflow.models.CategoricalModel.predict`
    * :func:`~probflow.models.CategoricalModel.metric`
    * :func:`~probflow.models.CategoricalModel.posterior_mean`
    * :func:`~probflow.models.CategoricalModel.posterior_sample`
    * :func:`~probflow.models.CategoricalModel.posterior_plot`
    * :func:`~probflow.models.CategoricalModel.prior_sample`
    * :func:`~probflow.models.CategoricalModel.prior_plot`
    * :func:`~probflow.models.CategoricalModel.log_prob`
    * :func:`~probflow.models.CategoricalModel.log_prob_by`
    * :func:`~probflow.models.CategoricalModel.prob`
    * :func:`~probflow.models.CategoricalModel.prob_by`
    * :func:`~probflow.models.CategoricalModel.summary`

    and adds the following categorical-model-specific methods:

    * :func:`~probflow.models.CategoricalModel.pred_dist_plot`
    * :func:`~probflow.models.CategoricalModel.calibration_curve`

    """


    def pred_dist_plot(self, 
                       x,
                       n=10000,
                       style='fill',
                       cols=1,
                       bins=20,
                       ci=0.0,
                       bw=0.075,
                       color=None,
                       alpha=0.4,
                       individually=False):
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
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns (if 
            ``individually=True``.
        bins : int or list or |ndarray|
            Number of bins to use for the posterior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram of the density
        individually : bool
            If ``True``, plot one subplot per datapoint in ``x``, otherwise
            plot all the predictive distributions on the same plot.
        """

        # Sample from the predictive distribution
        pred_samples = self.predictive_sample(x, n=n)

        # TODO


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
        #TODO
        pass
    


def save_model(model, filename):
    """Save a model to file"""
    pass 
    # TODO



def load_model(filename):
    """Load a model from file"""
    pass 
    # TODO

