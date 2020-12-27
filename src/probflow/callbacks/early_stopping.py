import numpy as np

from .callback import Callback
from .monitor_elbo import MonitorELBO
from .monitor_metric import MonitorMetric


class EarlyStopping(Callback):
    """Stop training early when some metric stops decreasing


    Parameters
    ----------
    metric_fn : callable, MonitorMetric, or MonitorELBO
        Any arbitrary function, or a :class:`.MonitorMetric` or
        :class:`.MonitorELBO` callback.  Training will be stopped when the
        value returned by that function stops decreasing (or, if ``metric_fn``
        was a :class:`.MonitorMetric` or :class:`.MonitorELBO`, training is
        stopped when the metric being monitored or the ELBO stops decreasing.
    patience : int
        Number of epochs to allow training to continue even if metric is not
        decreasing.  Default is 0.
    restore_best_weights : bool
        Whether or not to restore the weights from the best epoch after
        training is stopped.  Default = False.
    verbose : bool
        Whether to print when training was stopped.  Default = False
    name : str
        Name for this callback


    Example
    -------

    See the user guide section on :ref:`user-guide-early-stopping`.

    """

    def __init__(
        self, metric_fn, patience=0, verbose=True, name="EarlyStopping"
    ):

        # Check types
        if not isinstance(patience, int):
            raise TypeError("patience must be an int")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if not callable(metric_fn) and not isinstance(
            metric_fn, (MonitorMetric, MonitorELBO)
        ):
            raise TypeError(
                "metric_fn must be a callable, MonitorMetric, or MonitorELBO"
            )

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
        if isinstance(self.metric_fn, MonitorMetric):
            metric = self.metric_fn.current_metric
        elif isinstance(self.metric_fn, MonitorELBO):
            metric = self.metric_fn.current_elbo
        else:
            metric = self.metric_fn()
        if metric < self.best:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
            if self.count > self.patience:
                self.model.stop_training()
                if self.verbose:
                    print(self.name + " after " + str(self.epoch) + " epochs")
