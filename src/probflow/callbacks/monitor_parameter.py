import matplotlib.pyplot as plt

from .callback import Callback


class MonitorParameter(Callback):
    """Monitor the mean value of Parameter(s) over the course of training


    Parameters
    ----------
    params : str or List[str] or None
        Name(s) of the parameters to monitor.  If None, all parameters will be monitored.


    Examples
    --------

    See the user guide section on :ref:`user-guide-monitor-parameter`.

    """

    def __init__(self, params: str | list[str] | None):

        # Store metrics and epochs
        self.params: list[str] | None = (
            [params] if isinstance(params, str) else params
        )
        self.current_params: dict[str, float] | None = None
        self.current_epoch: int = 0
        self.parameter_values: list[dict[str, float]] = []
        self.epochs: list[int] = []

    def on_epoch_end(self) -> None:
        """Store mean values of Parameter(s) at the end of each epoch."""
        self.current_params = self.model.posterior_mean(self.params)
        self.current_epoch += 1
        self.parameter_values += [self.current_params]
        self.epochs += [self.current_epoch]

    def plot(self, param: str | list[str] | None = None, **kwargs) -> None:
        """Plot the parameter value(s) as a function of epoch

        Parameters
        ----------
        param : str or List[str] or None
            Parameter(s) to plot.  If None, plot all the monitored parameter(s).
        """
        # Check if parameter values have been recorded
        if self.parameter_values is None:
            raise RuntimeError("No parameter values have yet been recorded.")

        # Determine which parameters to plot
        params_to_plot: list[str]
        if isinstance(param, str):
            params_to_plot = [param]
        elif isinstance(param, list):
            params_to_plot = param
        else:
            params_to_plot = list(self.parameter_values[0].keys())

        # Plot the parameter values
        for i, param_name in enumerate(params_to_plot):
            plt.subplot(len(params_to_plot), 1, i + 1)
            plt.plot(
                self.epochs,
                [p[param_name] for p in self.parameter_values],
                **kwargs,
            )
            plt.xlabel("Epoch")
            plt.ylabel(f"{param_name} mean")
