import numpy as np

from .callback import Callback


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

    def __init__(
        self, metric_fn, patience=0, verbose=True, name="EarlyStopping"
    ):

        # Check types
        if not isinstance(patience, int):
            raise TypeError("patience must be an int")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if not callable(metric_fn):
            raise TypeError("metric_fn must be a callable")

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
                    print(self.name + " after " + str(self.epoch) + " epochs")
