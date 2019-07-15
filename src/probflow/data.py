"""Data utilities.

TODO: more info...

----------

"""



import numpy as np
import pandas as pd

from probflow.core.base import BaseDataGenerator



def train_val_split(x, y, val_split, val_shuffle):
    """Split data into training and validation data

    Parameters
    ----------
    x : |ndarray|, |DataFrame|, or |Series|
        Independent variable values.
    y : |ndarray|, |DataFrame|, or |Series|
        Dependent variable values.
    val_split : float between 0 and 1
        Proportion of the data to use as validation data.
    val_shuffle : bool
        Whether to shuffle which data is used for validation.  If False,
        the last ``val_split`` proportion of the input data is used
        for validation.

    Returns
    -------
    (N, x_train, y_train, x_val, y_val)

        * N: number of training samples
        * x_train: independent variable values of the training data
        * y_train: dependent variable values of the training data
        * x_val: independent variable values of the validation data
        * y_val: dependent variable values of the validation data
    """
    if val_split > 0:
        num_val = int(val_split*x.shape[0])
        train_ix = np.full(x.shape[0], True)
        train_ix[-num_val:] = False
        if val_shuffle:
            train_ix = np.random.permutation(train_ix)
        val_ix = ~train_ix
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x_train = x[train_ix]
            x_val = x[val_ix]
        else:
            x_train = x[train_ix, ...]
            x_val = x[val_ix, ...]
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_train = y[train_ix]
            y_val = y[val_ix]
        else:
            y_train = y[train_ix, ...]
            y_val = y[val_ix, ...]
    else:
        x_train = x
        y_train = y
        x_val = x
        y_val = y
    return x_train.shape[0], x_train, y_train, x_val, y_val



class DataGenerator(BaseDataGenerator):
    """Generate data to feed through a model.

    TODO

    """

    def __init__(self, x, y, batch_size=128, shuffle=True):

        # Check types
        if not isinstance(x, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError('x must be an ndarray, a DataFrame, or a Series')
        if not isinstance(y, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError('y must be an ndarray, a DataFrame, or a Series')
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be an int')
        if batch_size < 1:
            raise ValueError('batch_size must be >0')
        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be True or False')

        # Check sizes are consistent
        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y must contain same number of samples')

        # Store references to data
        self._batch_size = batch_size
        self.n_batches = int(np.ceil(x.shape[0]/batch_size))
        self.x = x
        self.y = y

        # Shuffle data
        self.shuffle = shuffle
        self.on_epoch_end()
        

    @property
    def n_samples(self):
        """Number of samples in the dataset"""
        return self.x.shape[0]


    @property
    def batch_size(self):
        """Number of samples per batch"""
        return self._batch_size


    def __getitem__(self, index):
        """Generate one batch of data"""

        # Get shuffled indexes
        ix = self.ids[index*self.batch_size:(index+1)*self.batch_size]

        # Get x data
        if isinstance(self.x, pd.DataFrame):
            x = self.x.iloc[ix, :]
        elif isinstance(self.x, pd.Series):
            x = self.x.iloc[ix]
        else:
            x = self.x[ix, ...]

        # Get y data
        if isinstance(self.y, pd.DataFrame):
            x = self.y.iloc[ix, :]
        elif isinstance(self.y, pd.Series):
            x = self.y.iloc[ix]
        else:
            y = self.y[ix, ...]

        # Return both x and y
        return x, y


    def on_epoch_end(self):
        """Shuffle data each epoch"""
        if self.shuffle:
            self.ids = np.random.permutation(self.n)
        else:
            self.ids = np.arange(self.n, dtype=np.uint64)