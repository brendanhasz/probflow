import time

import matplotlib.pyplot as plt
import numpy as np

from .callback import Callback


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
        self.start_time = None
        self.wall_times = []

    def on_epoch_start(self):
        """Record start time at the beginning of the first epoch"""
        if self.start_time is None:
            self.start_time = time.time()

    def on_epoch_end(self):
        """Store the ELBO at the end of each epoch."""
        self.current_elbo = self.model.get_elbo()
        self.current_epoch += 1
        self.elbos += [self.current_elbo]
        self.epochs += [self.current_epoch]
        self.wall_times += [time.time() - self.start_time]
        if self.verbose:
            print(
                "Epoch {} \tELBO: {}".format(
                    self.current_epoch, self.current_elbo
                )
            )

    def plot(self, x="epoch", **kwargs):
        """Plot the ELBO as a function of epoch

        Parameters
        ----------
        x : str {'epoch' or 'time'}
            Whether to plot the metric as a function of epoch or wall time
            Default is to plot by epoch.
        **kwargs
            Additional keyword arguments are passed to plt.plot
        """
        if x == "time":
            plt.plot(self.wall_times, self.elbos, **kwargs)
            plt.xlabel("Time (s)")
        else:
            plt.plot(self.epochs, self.elbos, **kwargs)
            plt.xlabel("Epoch")
        plt.ylabel("ELBO")
