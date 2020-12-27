import time

import matplotlib.pyplot as plt
import numpy as np

from probflow.data import make_generator
from probflow.utils.metrics import get_metric_fn

from .callback import Callback


class MonitorMetric(Callback):
    """Monitor some metric on validation data


    Parameters
    ----------
    metric : str
        Name of the metric to evaluate.  See :meth:`.Model.metric` for a list
        of available metrics.
    x : |ndarray| or |DataFrame| or |Series| or Tensor or |DataGenerator|
        Independent variable values of the validation dataset to evaluate (aka
        the "features").  Or a |DataGenerator| to generate both x and y.
    y : |ndarray| or |DataFrame| or |Series| or Tensor
        Dependent variable values of the validation dataset to evaluate (aka
        the "target").
    verbose : bool
        Whether to print the average ELBO at the end of every training epoch
        (if True) or not (if False).  Default = False


    Example
    -------

    See the user guide section on :ref:`monitoring-a-metric`.
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
        self.start_time = None
        self.wall_times = []

    def on_epoch_start(self):
        """Record start time at the beginning of the first epoch"""
        if self.start_time is None:
            self.start_time = time.time()

    def on_epoch_end(self):
        """Compute the metric on validation data at the end of each epoch."""
        self.current_metric = self.model.metric(self.metric_fn, self.data)
        self.current_epoch += 1
        self.metrics += [self.current_metric]
        self.epochs += [self.current_epoch]
        self.wall_times += [time.time() - self.start_time]
        if self.verbose:
            print(
                "Epoch {} \t{}: {}".format(
                    self.current_epoch, self.metric_name, self.current_metric
                )
            )

    def plot(self, x="epoch", **kwargs):
        """Plot the metric being monitored as a function of epoch

        Parameters
        ----------
        x : str {'epoch' or 'time'}
            Whether to plot the metric as a function of epoch or wall time.
            Default is to plot by epoch.
        **kwargs
            Additional keyword arguments are passed to plt.plot
        """
        if x == "time":
            plt.plot(self.wall_times, self.metrics, **kwargs)
            plt.xlabel("Time (s)")
        else:
            plt.plot(self.epochs, self.metrics, **kwargs)
            plt.xlabel("Epoch")
        plt.ylabel(self.metric_name)
