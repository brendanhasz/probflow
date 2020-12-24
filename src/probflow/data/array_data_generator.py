import numpy as np
import pandas as pd

from .data_generator import DataGenerator


class ArrayDataGenerator(DataGenerator):
    """Generate array-structured data to feed through a model.

    TODO

    Parameters
    ----------
    x : |ndarray| or |DataFrame| or |Series| or |DataGenerator|
        Independent variable values (or, if fitting a generative model,
        the dependent variable values).  Should be of shape (Nsamples,...)
    y : |None| or |ndarray| or |DataFrame| or |Series|
        Dependent variable values (or, if fitting a generative model,
        ``None``). Should be of shape (Nsamples,...).  Default = ``None``
    batch_size : int
        Number of samples to use per minibatch.  Use ``None`` to use a single
        batch for all the data.
        Default = ``None``
    shuffle : bool
        Whether to shuffle the data each epoch.
        Default = ``False``
    testing : bool
        Whether to treat data as testing data (allow no dependent variable).
        Default = ``False``
    """

    def __init__(
        self,
        x=None,
        y=None,
        batch_size=None,
        shuffle=False,
        test=False,
        num_workers=None,
    ):

        # Set number of worker threads
        super().__init__(num_workers=num_workers)

        # Check types
        data_types = (np.ndarray, pd.DataFrame, pd.Series)
        if x is not None and not isinstance(x, data_types):
            raise TypeError("x must be an ndarray, a DataFrame, or a Series")
        if y is not None and not isinstance(y, data_types):
            raise TypeError("y must be an ndarray, a DataFrame, or a Series")
        if batch_size is not None:
            if not isinstance(batch_size, int):
                raise TypeError("batch_size must be an int")
            if batch_size < 1:
                raise ValueError("batch_size must be >0")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False")
        if not isinstance(test, bool):
            raise TypeError("test must be True or False")

        # No data? (eg when sampling from a generative model)
        if x is None and y is None:
            self._empty = True
            self._n_samples = 1
            self._batch_size = 1
            return
        else:
            self._empty = False

        # Check sizes are consistent
        if x is not None and y is not None:
            if x.shape[0] != y.shape[0]:
                raise ValueError("x and y must contain same number of samples")

        # Generative model?
        if not test and y is None:
            y = x
            x = None

        # Number of samples
        if x is None:
            self._n_samples = y.shape[0]
        else:
            self._n_samples = x.shape[0]

        # Batch size
        if batch_size is None or self._n_samples < batch_size:
            self._batch_size = self._n_samples
        else:
            self._batch_size = batch_size

        # Store references to data
        self.x = x
        self.y = y

        # Shuffle data
        self.shuffle = shuffle
        self.on_epoch_end()

    @property
    def n_samples(self):
        """Number of samples in the dataset"""
        return self._n_samples

    @property
    def batch_size(self):
        """Number of samples to generate each minibatch"""
        return self._batch_size

    def get_batch(self, index):
        """Generate one batch of data"""

        # Return none if no data
        if self._empty:
            return None, None

        # Get shuffled indexes
        ix = slice(index * self.batch_size, (index + 1) * self.batch_size)
        if self.shuffle:
            ix = self.ids[ix]

        # Get x data
        if self.x is None:
            x = None
        elif isinstance(self.x, pd.DataFrame):
            x = self.x.iloc[ix, :]
        elif isinstance(self.x, pd.Series):
            x = self.x.iloc[ix]
        else:
            x = self.x[ix, ...]

        # Get y data
        if self.y is None:
            y = None
        elif isinstance(self.y, pd.DataFrame):
            y = self.y.iloc[ix, :]
        elif isinstance(self.y, pd.Series):
            y = self.y.iloc[ix]
        else:
            y = self.y[ix, ...]

        # Return both x and y
        return x, y

    def on_epoch_end(self):
        """Shuffle data each epoch"""
        if self.shuffle:
            self.ids = np.random.permutation(self.n_samples)
