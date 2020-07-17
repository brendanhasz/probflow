from .callback import Callback


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
