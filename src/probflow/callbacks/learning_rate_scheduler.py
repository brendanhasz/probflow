import matplotlib.pyplot as plt

from .callback import Callback


class LearningRateScheduler(Callback):
    """Set the learning rate as a function of the current epoch

    Parameters
    ----------
    fn : callable
        Function which takes the current epoch as an argument and returns a
        learning rate.


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
        self.current_lr = 0
        self.epochs = []
        self.learning_rate = []

    def on_epoch_start(self):
        """Set the learning rate at the start of each epoch."""
        self.current_epoch += 1
        self.current_lr = self.fn(self.current_epoch)
        self.model.set_learning_rate(self.current_lr)
        self.epochs += [self.current_epoch]
        self.learning_rate += [self.current_lr]

    def plot(self):
        plt.plot(self.epochs, self.learning_rate)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
