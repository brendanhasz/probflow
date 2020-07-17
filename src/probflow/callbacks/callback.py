from probflow.utils.base import BaseCallback


class Callback(BaseCallback):
    """

    TODO

    """

    def __init__(self, *args):
        """Initialize the callback"""

    def on_epoch_start(self):
        """Will be called at the start of each training epoch.  By default does
        nothing."""

    def on_epoch_end(self):
        """Will be called at the end of each training epoch.  By default does
        nothing."""

    def on_train_end(self):
        """Will be called at the end of training. By default does nothing."""
