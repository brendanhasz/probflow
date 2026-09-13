"""Stop training after a certain amount of time."""

import time

from .callback import Callback


class TimeOut(Callback):
    """Stop training after a certain amount of time.

    Parameters
    ----------
    time_limit : float or int
        Number of seconds after which to stop training
    verbose : bool
        Whether to print that we stopped training early (if True) or not (if
        False).  Default = False


    Example
    -------

    Stop training after five hours:

    .. code-block:: python3

        time_out = pf.callbacks.TimeOut(5*60*60)
        model.fit(x, y, callbacks=[time_out])

    """

    def __init__(self, time_limit: float, verbose: bool = True):

        # Store values
        self.time_limit: float | int = time_limit
        self.start_time: float | None = None
        self.verbose: bool = verbose

    def on_epoch_start(self) -> None:
        """Record start time at the beginning of the first epoch."""
        if self.start_time is None:
            self.start_time = time.time()

    def on_epoch_end(self) -> None:
        """Stop training if time limit has been passed."""
        if self.start_time is None:
            raise RuntimeError(
                "TimeOut callback was not initialized properly.  "
                "on_epoch_start() was not called before on_epoch_end()."
            )
        dt = time.time() - self.start_time
        if self.time_limit < dt:
            self.model.stop_training()
            if self.verbose:
                print(f"TimeOut callback ended training after {dt} s")
