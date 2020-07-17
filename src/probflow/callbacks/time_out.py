import time

from .callback import Callback


class TimeOut(Callback):
    """Stop training after a certain amount of time

    TODO

    Parameters
    ----------
    time_limit : float or int
        Number of seconds after which to stop training

    Example
    -------

    Stop training after five hours:

    .. code-block:: python3

        time_out = pf.callbacks.TimeOut(5*60*60)
        model.fit(x, y, callbacks=[time_out])

    """

    def __init__(self, time_limit, verbose=True):

        # Store values
        self.time_limit = time_limit
        self.start_time = None
        self.verbose = verbose

    def on_epoch_end(self):
        """Stop training if time limit has been passed"""
        if self.start_time is None:
            self.start_time = time.time()
        dt = time.time() - self.start_time
        if self.time_limit < dt:
            self.model.stop_training()
            if self.verbose:
                print("TimeOut after " + str(dt) + "s")
