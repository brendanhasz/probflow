import matplotlib.pyplot as plt

from .callback import Callback


class MonitorParameter(Callback):
    """Monitor the mean value of Parameter(s) over the course of training


    Parameters
    ----------
    params : str or List[str] or None
        Name(s) of the parameters to monitor.


    Examples
    --------

    See the user guide section on :ref:`user-guide-monitor-parameter`.

    """

    def __init__(self, params):

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

    def plot(self, param=None, **kwargs):
        """Plot the parameter value(s) as a function of epoch

        Parameters
        ----------
        param : None or str
            Parameter to plot.  If None, assumes we've only been monitoring one
            parameter and plots that.  If a str, plots the parameter with that
            name (assuming we've been monitoring it).
        """
        if param is None:  # assume we've only been monitoring one parameter
            plt.plot(self.epochs, self.parameter_values, **kwargs)
            plt.xlabel("Epoch")
            plt.ylabel(f"{self.params} mean")
        else:  # plot a specific parameter
            plt.plot(
                self.epochs,
                [p[param] for p in self.parameter_values],
                **kwargs,
            )
            plt.xlabel("Epoch")
            plt.ylabel(f"{param} mean")
