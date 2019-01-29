"""Data utilities.

TODO: more info...

----------

"""

import numpy as np



def process_data(model, x=None, data=None, name='x'):
    """Process and validate just the x data"""

    # Ensure we were passed a model object
    # TODO

    # Use training data if none passed
    if x is None:
        try:
            x = model._train[name]
        except AttributeError:
            raise RuntimeError(name+' cannot be None if model is not fit')

    # Numpy arrays
    if data is None:

        # Ensure data is numpy array
        if not isinstance(x, np.ndarray):
            raise TypeError(name+' must be a numpy ndarray')

    else:

        # Pandas DataFrame
        from pandas import DataFrame
        if isinstance(data, DataFrame):

            # Check type
            if not isinstance(x, (int, str, list)):
                raise TypeError(name+' must be an int, string, or list')
            if isinstance(x, list):
                if not all([isinstance(e, (int, str)) for e in x]):
                    raise TypeError(name+' must be a list of int or str')

            # Get the columns
            x = data.ix[:,x].values

        else:
            raise TypeError('data must be None or a pandas DataFrame')

    # Make data at least 2 dimensional (0th dim should be N)
    if x.ndim == 1:
        x = np.expand_dims(x, 1)

    # TODO: ensure x data shape matches model._ph['x'] shape (only if fit)

    return x


def process_xy_data(self, x=None, y=None, data=None):
    """Process and validate both x and y data"""

    # Both or neither of x and y should be passed
    if x is None and y is not None or y is None and x is not None:
        raise TypeError('x and y should both be set or both be None')

    # Process both x and y data
    x = process_data(self, x, data, name='x')
    y = process_data(self, y, data, name='y')

    return x, y


def test_train_split(x, y, val_split, val_shuffle):
    """Split data into training and validation data"""
    if val_split > 0:
        if val_shuffle:
            train_ix = np.random.rand(x.shape[0]) > val_split
        else:
            num_val = int(val_split*x.shape[0])
            train_ix = np.full(x.shape[0], True)
            train_ix[-num_val:] = False
        val_ix = ~train_ix
        x_train = x[train_ix, ...]
        y_train = y[train_ix, ...]
        x_val = x[val_ix, ...]
        y_val = y[val_ix, ...]
    else:
        x_train = x
        y_train = y
        x_val = x
        y_val = y
    return x_train.shape[0], x_train, y_train, x_val, y_val
