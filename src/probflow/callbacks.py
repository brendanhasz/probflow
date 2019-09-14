"""
The callbacks module contains classes for monitoring and adjusting the 
training process.

* :class:`.Callback` - abstract base class for all callbacks
* :class:`.LearningRateScheduler` - set the learning rate by epoch
* :class:`.MonitorMetric` - record a metric over the course of training
* :class:`.MonitorParameter` - record a parameter over the course of training
* :class:`.EarlyStopping` - stop training if some metric stops improving

----------

"""


__all__ = [
    'Callback',
    'LearningRateScheduler',
    'MonitorMetric',
    'MonitorParameter',
    'EarlyStopping',
]



import numpy as np

from probflow.core.base import BaseCallback
from probflow.data import DataGenerator
from probflow.data import make_generator
from probflow.utils.metrics import get_metric_fn



class Callback(BaseCallback):
    """

    TODO

    """
    
    def __init__(self, *args):
        pass


    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""
        pass


    def on_train_end(self):
        """Will be called at the end of training"""
        pass



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


    def on_epoch_end(self):
        """Set the learning rate at the end of each epoch"""
        self.current_epoch += 1
        self.current_lr = self.fn(self.current_epoch)
        self.model.set_learning_rate(self.current_lr)
        self.epochs += [self.current_epoch]
        self.learning_rate += [self.current_lr]



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

    def __init__(self, metric, x, y=None, verbose=True):

        # Store metric
        self.metric_fn = get_metric_fn(metric)

        # Store validation data
        self.data = make_generator(x, y)

        # Store metrics and epochs
        self.current_metric = np.nan
        self.current_epoch = 0
        self.metrics = []
        self.epochs = []
        self.verbose = verbose


    def on_epoch_end(self):
        """Compute metric on validation data"""
        self.current_metric = self.model.metric(self.metric_fn, self.data)
        self.current_epoch += 1
        self.metrics += [self.current_metric]
        self.epochs += [self.current_epoch]
        if self.verbose:
            print('Epoch {} \t{}: {}'.format(
                  self.current_epoch,
                  self.metric_fn.__name__,
                  self.current_metric))



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
        """Store mean values of Parameter(s) each epoch"""
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
    
    def __init__(self, metric_fn, patience=0):

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
        # TODO: restore_best_weights? using save_model and load_model?


    def on_epoch_end(self):
        """Will be called at the end of each training epoch"""
        metric = self.metric_fn()
        if metric < self.best:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count > self.patience:
                self.model.stop_training()
