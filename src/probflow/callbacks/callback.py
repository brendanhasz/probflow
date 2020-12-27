from probflow.utils.base import BaseCallback


class Callback(BaseCallback):
    """Base class for all callbacks.

    See the user guide section on :ref:`user_guide_callbacks`.

    """

    def __init__(self, *args):
        """Initialize the callback"""

    def on_train_start(self):
        """Will be called at the start of training. By default does nothing."""

    def on_epoch_start(self):
        """Will be called at the start of each training epoch.  By default does
        nothing."""

    def on_epoch_end(self):
        """Will be called at the end of each training epoch.  By default does
        nothing."""

    def on_train_end(self):
        """Will be called at the end of training. By default does nothing."""
