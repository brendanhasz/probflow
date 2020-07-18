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

    def on_epoch_end(self):
        """Store the ELBO at the end of each epoch."""
        self.current_elbo = self.model.get_elbo()
        self.current_epoch += 1
        self.elbos += [self.current_elbo]
        self.epochs += [self.current_epoch]
        if self.verbose:
            print(
                "Epoch {} \tELBO: {}".format(
                    self.current_epoch, self.current_elbo
                )
            )

    def plot(self):
        plt.plot(self.epochs, self.elbos)
        plt.xlabel("Epoch")
        plt.ylabel("ELBO")
