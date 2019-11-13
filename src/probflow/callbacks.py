"""
The callbacks module contains classes for monitoring and adjusting the 
training process.

* :class:`.Callback` - abstract base class for all callbacks
* :class:`.LearningRateScheduler` - set the learning rate by epoch
* :class:`.KLWeightScheduler` - set the KL weight by epoch
* :class:`.MonitorMetric` - record a metric over the course of training
* :class:`.MonitorParameter` - record a parameter over the course of training
* :class:`.EarlyStopping` - stop training if some metric stops improving
* :class:`.TimeOut` - stop training after a certain amount of time

----------

"""


__all__ = [
    'Callback',
    'LearningRateScheduler',
    'KLWeightScheduler',
    'MonitorMetric',
    'MonitorELBO',
    'MonitorParameter',
    'EarlyStopping',
    'TimeOut',
]



import time

import numpy as np
import matplotlib.pyplot as plt

from probflow.core.base import BaseCallback
from probflow.data import DataGenerator
from probflow.data import make_generator
from probflow.utils.metrics import get_metric_fn



class Callback(BaseCallback):
    """

    TODO

    """
    
    def __init__(self, *args):
        """Initialize the callback"""


    def on_epoch_start(self):
        """Will be called at the start of each training epoch.  By default does
        nothing."""


    def on_epoch_end(self):
        """Will be called at the end of each training epoch.  By default does
        nothing."""


    def on_train_end(self):
        """Will be called at the end of training. By default does nothing."""



class LearningRateScheduler(Callback):
    """Set the learning rate as a function of the current epoch

    Parameters
    ----------
    fn : callable
        Function which takes the current epoch as an argument and returns a 
        learning rate.


    Examples
    --------

    TODO
    """
    
    def __init__(self, fn):
        
        # Check type
        if not callable(fn):
            raise TypeError('fn must be a callable')
        if not isinstance(fn(1), float):
            raise TypeError('fn must return a float')

        # Store function
        self.fn = fn
        self.current_epoch = 0
        self.current_lr = 0
        self.epochs = []
        self.learning_rate = []


    def on_epoch_start(self):
        """Set the learning rate at the start of each epoch."""
        self.current_epoch += 1
        self.current_lr = self.fn(self.current_epoch)
        self.model.set_learning_rate(self.current_lr)
        self.epochs += [self.current_epoch]
        self.learning_rate += [self.current_lr]


    def plot(self):
        plt.plot(self.epochs, self.learning_rate)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')



class KLWeightScheduler(Callback):
    """Set the weight of the KL term's contribution to the ELBO loss each epoch

    Parameters
    ----------
    fn : callable
        Function which takes the current epoch as an argument and returns a 
        kl weight, a float between 0 and 1


    Examples
    --------

    TODO
    """
    
    def __init__(self, fn):
        
        # Check type
        if not callable(fn):
            raise TypeError('fn must be a callable')
        if not isinstance(fn(1), float):
            raise TypeError('fn must return a float')

        # Store function
        self.fn = fn
        self.current_epoch = 0
        self.current_w = 0
        self.epochs = []
        self.kl_weights = []


    def on_epoch_start(self):
        """Set the KL weight at the start of each epoch."""
        self.current_epoch += 1
        self.current_w = self.fn(self.current_epoch)
        self.model.set_kl_weight(self.current_w)
        self.epochs += [self.current_epoch]
        self.kl_weights += [self.current_w]


    def plot(self):
        plt.plot(self.epochs, self.kl_weights)
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss Weight')



class MonitorMetric(Callback):
    """Monitor some metric on validation data

    TODO: docs

    Example
    -------

    To record the mean absolute error of a model over the course of training,
    we can create a :class:`.MonitorMetric` callback:

    .. code-block:: python3

        #x_val and y_val are numpy arrays w/ validation data
        monitor_mae = MonitorMetric('mse', x_val, y_val)

        model.fit(x_train, y_train, callbacks=[monitor_mae])
    """

    def __init__(self, metric, x, y=None, verbose=False):

        # Store metric
        self.metric_fn = get_metric_fn(metric)
        if isinstance(metric, str):
            self.metric_name = metric
        else:
            self.metric_name = self.metric_fn.__name__

        # Store validation data
        self.data = make_generator(x, y)

        # Store metrics and epochs
        self.current_metric = np.nan
        self.current_epoch = 0
        self.metrics = []
        self.epochs = []
        self.verbose = verbose


    def on_epoch_end(self):
        """Compute the metric on validation data at the end of each epoch."""
        self.current_metric = self.model.metric(self.metric_fn, self.data)
        self.current_epoch += 1
        self.metrics += [self.current_metric]
        self.epochs += [self.current_epoch]
        if self.verbose:
            print('Epoch {} \t{}: {}'.format(
                  self.current_epoch,
                  self.metric_name,
                  self.current_metric))


    def plot(self):
        plt.plot(self.epochs, self.metrics)
        plt.xlabel('Epoch')
        plt.ylabel(self.metric_name)



class MonitorELBO(Callback):
    """Monitor the ELBO on the training data

    TODO: docs

    Example
    -------

    To record the evidence lower bound (ELBO) for each batch of training data
    over the course of training, we can create a :class:`.MonitorELBO`
    callback:

    .. code-block:: python3

        monitor_elbo = MonitorELBO()

        model.fit(x_train, y_train, callbacks=[monitor_elbo])
    """

    def __init__(self, verbose=False):
        self.current_elbo = np.nan
        self.current_epoch = 0
        self.elbos = []
        self.epochs = []
        self.verbose = verbose


    def on_epoch_end(self):
        """Store the ELBO at the end of each epoch."""
        self.current_elbo = self.model.get_elbo()
        self.current_epoch += 1
        self.elbos += [self.current_elbo]
        self.epochs += [self.current_epoch]
        if self.verbose:
            print('Epoch {} \tELBO: {}'.format(
                  self.current_epoch,
                  self.current_elbo))


    def plot(self):
        plt.plot(self.epochs, self.elbos)
        plt.xlabel('Epoch')
        plt.ylabel('ELBO')



class MonitorParameter(Callback):
    """Monitor the mean value of Parameter(s) over the course of training

    TODO

    """

    def __init__(self, x, y=None, params=None):

        # Store metrics and epochs
        self.params = params
        self.current_params = None
        self.current_epoch = 0
        self.parameter_values = []
        self.epochs = []


    def on_epoch_end(self):
        """Store mean values of Parameter(s) at the end of each epoch."""
        self.current_params = self.model.posterior_mean(self.params)
        self.current_epoch += 1
        self.parameter_values += [self.current_params]
        self.epochs += [self.current_epoch]



class EarlyStopping(Callback):
    """Stop training early when some metric stops decreasing

    TODO

    Example
    -------

    Stop training when the mean absolute error stops improving, we can create
    a :class:`.EarlyStopping` callback which monitors the current value of
    the MAE via a :class:`.MonitorMetric` callback:

    .. code-block:: python3

        #x_val and y_val are numpy arrays w/ validation data
        monitor_mae = MonitorMetric('mse', x_val, y_val)
        early_stopping = EarlyStopping(lambda: monitor_mae.current_metric)

        model.fit(x_train, y_train, callbacks=[monitor_mae, early_stopping])

    """
    
    def __init__(self, metric_fn, patience=0, verbose=True, 
                 name='EarlyStopping'):

        # Check types
        if not isinstance(patience, int):
            raise TypeError('patience must be an int')
        if patience < 0:
            raise ValueError('patience must be non-negative')
        if not callable(metric_fn):
            raise TypeError('metric_fn must be a callable')

        # Store values
        self.metric_fn = metric_fn
        self.patience = patience
        self.best = np.Inf
        self.count = 0
        self.epoch = 0
        self.verbose = verbose
        self.name = name
        # TODO: restore_best_weights? using save_model and load_model?


    def on_epoch_end(self):
        """Stop training if there was no improvement since the last epoch."""
        self.epoch += 1
        metric = self.metric_fn()
        if metric < self.best:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count > self.patience:
                self.model.stop_training()
                if self.verbose:
                    print(self.name+' after '+str(self.epoch)+' epochs')



class TimeOut(Callback):
    """Stop training after a certain amount of time

    TODO

    Parameters
    ----------
    time_limit : float or int
        Number of seconds after which to stop training

    Example
    -------

    Stop training after five hours:

    .. code-block:: python3

        time_out = pf.callbacks.TimeOut(5*60*60)
        model.fit(x, y, callbacks=[time_out])

    """
    
    def __init__(self, time_limit, verbose=True):

        # Store values
        self.time_limit = time_limit
        self.start_time = None
        self.verbose = verbose


    def on_epoch_end(self):
        """Stop training if time limit has been passed"""
        if self.start_time is None:
            self.start_time = time.time()
        dt = time.time()-self.start_time
        if self.time_limit < dt:
            self.model.stop_training()
            if self.verbose:
                print('TimeOut after '+str(dt)+'s')

