.. _example_nmf:

Neural Matrix Factorization
---------------------------

.. include:: macros.hrst


TODO: description...

TODO: for a vanilla matrix factorization

.. code-block:: python

    from probflow import *

    #df = DataFrame w/ 3 columns: 'user_id', 'item_id', and 'rating'

    # User and item IDs
    users = Input('user_id')
    items = Input('item_id')

    # Matrix Factorization
    user_vec = Embedding(users, dims=50)
    item_vec = Embedding(items, dims=50)
    predictions = Dot(user_vec, item_vec)
    error = ScaleParameter()

    # Fit a model w/ normally-distibuted error
    model = Normal(predictions, error)
    model.fit(df[['user_id', 'item_id']], df['rating'])

or for neural matrix factorization https://arxiv.org/abs/1708.05031

.. code-block:: python

    from probflow import *

    #df = DataFrame w/ 3 columns: 'user_id', 'item_id', and 'rating'

    # User and item IDs
    users = Input('user_id')
    items = Input('item_id')

    # Matrix Factorization
    user_vec_mf = Embedding(users, dims=50)
    item_vec_mf = Embedding(items, dims=50)
    predictions_mf = Dot(user_vec_mf, item_vec_mf)

    # Neural Collaborative Filtering
    user_vec_ncf = Embedding(users, dims=50)
    item_vec_ncf = Embedding(items, dims=50)
    ncf_in = Cat([user_vec_ncf, item_vec_ncf])
    predictions_ncf = DenseNet(ncf_in, units=[128, 64, 32])
    
    # Combine the two methods
    predictions = Dense(Cat([predictions_mf, predictions_ncf]))
    error = ScaleParameter()

    # Fit a model w/ normally-distibuted error
    model = Normal(predictions, error)
    model.fit(df[['user_id', 'item_id']], df['rating'])

Or if you have implicit data (0 or 1 for whether the user has interacted with 
the item), use model = Bernoulli(logits=predictions)

Or if there are discrete scores (e.g. 1-10), then use a BetaBinomial 
TODO: w/ Dense and Embedding layers, then w/ NeuralMatrixFactorization