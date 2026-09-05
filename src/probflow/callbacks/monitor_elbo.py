"""Monitor the ELBO on the training data."""

import time

import matplotlib.pyplot as plt
import numpy as np

from .callback import Callback


class MonitorELBO(Callback):
    """Monitor the ELBO on the training data.

    Parameters
    ----------
    verbose : bool
        Whether to print the average ELBO at the end of every training epoch
        (if True) or not (if False).  Default = False


    Example
    -------

    See the user guide section on :ref:`monitoring-the-loss`.

    """

    def __init__(self, verbose: bool = False):
        self.current_elbo: float = np.nan
        self.current_epoch: int = 0
        self.elbos: list[float] = []
        self.epochs: list[int] = []
        self.verbose: bool = verbose
        self.start_time: float | None = None
        self.wall_times: list[float] = []

    def on_epoch_start(self) -> None:
        """Record start time at the beginning of the first epoch."""
        if self.start_time is None:
            self.start_time = time.time()

    def on_epoch_end(self) -> None:
        """Store the ELBO at the end of each epoch."""
        if self.start_time is None:
            raise RuntimeError(
                "MonitorELBO callback was not initialized properly.  "
                "on_epoch_start() was not called before on_epoch_end()."
            )
        self.current_elbo = self.model.get_elbo()
        self.current_epoch += 1
        self.elbos += [self.current_elbo]
        self.epochs += [self.current_epoch]
        self.wall_times += [time.time() - self.start_time]
        if self.verbose:
            print(f"Epoch {self.current_epoch} \tELBO: {self.current_elbo}")

    def plot(self, x: str = "epoch", **kwargs) -> None:
        """Plot the ELBO as a function of epoch.

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
