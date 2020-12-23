import matplotlib.pyplot as plt

from .callback import Callback


class KLWeightScheduler(Callback):
    """Set the weight of the KL term's contribution to the ELBO loss each epoch

    Parameters
    ----------
    fn : callable
        Function which takes the current epoch as an argument and returns a
        kl weight, a float between 0 and 1


    Examples
    --------

    TODO
    """

    def __init__(self, fn):

        # Check type
        if not callable(fn):
            raise TypeError("fn must be a callable")
        if not isinstance(fn(1), float):
            raise TypeError("fn must return a float")

        # Store function
        self.fn = fn
        self.current_epoch = 0
        self.current_w = 0
        self.epochs = []
        self.kl_weights = []

    def on_epoch_start(self):
        """Set the KL weight at the start of each epoch."""
        self.current_epoch += 1
        self.current_w = self.fn(self.current_epoch)
        self.model.set_kl_weight(self.current_w)
        self.epochs += [self.current_epoch]
        self.kl_weights += [self.current_w]

    def plot(self, **kwargs):
        """Plot the KL weight as a function of epoch

        Parameters
        ----------
        **kwargs
            Additional keyword arguments are passed to plt.plot
        """
        plt.plot(self.epochs, self.kl_weights, **kwargs)
        plt.xlabel("Epoch")
        plt.ylabel("KL Loss Weight")
