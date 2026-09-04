from collections.abc import Callable

import matplotlib.pyplot as plt

from .callback import Callback


class KLWeightScheduler(Callback):
    """Set the weight of the KL term's contribution to the ELBO loss each epoch

    Parameters
    ----------
    fn : Callable[[int], float]
        Function which takes the current epoch as an argument and returns a
        kl weight, a float between 0 and 1
    verbose : bool
        Whether to print the KL weight each epoch (if True) or not (if False).
        Default = False


    Examples
    --------

    See the user guide section on :ref:`user-guide-kl-scheduler`.

    """

    def __init__(self, fn: Callable[[int], float], verbose: bool = False):

        # Check type
        if not callable(fn):
            raise TypeError("fn must be a callable")
        if not isinstance(fn(1), float):
            raise TypeError("fn must return a float given an epoch number")

        # Store function
        self.fn: Callable[[int], float] = fn
        self.verbose: bool = verbose
        self.current_epoch: int = 0
        self.current_w: float = 0
        self.epochs: list[int] = []
        self.kl_weights: list[float] = []

    def on_epoch_start(self) -> None:
        """Set the KL weight at the start of each epoch."""
        self.current_epoch += 1
        self.current_w = self.fn(self.current_epoch)
        self.model.set_kl_weight(self.current_w)
        self.epochs += [self.current_epoch]
        self.kl_weights += [self.current_w]
        if self.verbose:
            print(f"Epoch {self.current_epoch} - KL weight {self.current_w}")

    def plot(self, **kwargs) -> None:
        """Plot the KL weight as a function of epoch

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed to plt.plot
        """
        plt.plot(self.epochs, self.kl_weights, **kwargs)
        plt.xlabel("Epoch")
        plt.ylabel("KL Loss Weight")
