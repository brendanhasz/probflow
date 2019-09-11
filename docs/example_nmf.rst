.. _example_nmf:

Neural Matrix Factorization
---------------------------

.. include:: macros.hrst


TODO: update this to v2

TODO: description...

TODO: for a vanilla matrix factorization

.. code-block:: python

    import probflow as pf

    class MatrixFactorization(pf.Model):

        def __init__(self, Nu, Ni, Nd):
            self.user_emb = pf.Embedding(Nu, Nd)
            self.item_emb = pf.Embedding(Ni, Nd)
            self.std = pf.ScaleParameter()

        def __call__(self, x):
            user_vec = self.user_emb(x['user_id'])
            item_vec = self.item_emb(x['item_id'])
            predictions = tf.matmul(user_vec, tf.transpose(item_vec))
            return pf.Normal(predictions, self.std())

TODO: Then can instantiate the model

.. code-block:: python

    #df = DataFrame w/ 3 columns: 'user_id', 'item_id', and 'rating'
    Nu = df['user_id'].nunique() #number of users
    Ni = df['item_id'].nunique() #number of items
    Nd = 50 #number of embedding dimensions
    model = MatrixFactorization(Nu, Ni, Nd)

TODO: Then fit it;

.. code-block:: python

    model.fit(df[['user_id', 'item_id']], df['rating'])


or for neural matrix factorization https://arxiv.org/abs/1708.05031

.. code-block:: python

    import probflow as pf

    class NeuralMatrixFactorization(pf.Model):

        def __init__(self, Nu, Ni, Nd, dims):
            self.user_mf = pf.Embedding(Nu, Nd)
            self.item_mf = pf.Embedding(Ni, Nd)
            self.user_ncf = pf.Embedding(Nu, Nd)
            self.item_ncf = pf.Embedding(Ni, Nd)
            self.net = pf.DenseNetwork(dims)
            self.linear = pf.Dense(dims[-1]+)
            self.std = pf.ScaleParameter()

        def __call__(self, x):
            user_mf = self.user_mf(x['user_id'])
            item_mf = self.item_mf(x['item_id'])
            user_ncf = self.user_ncf(x['user_id'])
            item_ncf = self.item_ncf(x['item_id'])
            preds_mf = user_mf*item_mf
            preds_ncf = self.net(tf.concat([user_ncf, item_ncf], -1))
            preds = self.linear(tf.concat([preds_mf, preds_ncf], -1))
            return pf.Normal(preds, self.std())


TODO: Then can instantiate the model

.. code-block:: python

    #df = DataFrame w/ 3 columns: 'user_id', 'item_id', and 'rating'
    Nu = df['user_id'].nunique() #number of users
    Ni = df['item_id'].nunique() #number of items
    Nd = 50 #number of embedding dimensions
    dims = [Nd*2, 128, 64, 32]
    model = NeuralMatrixFactorization(Nu, Ni, Nd, dims)

TODO: Then fit it;

.. code-block:: python

    model.fit(df[['user_id', 'item_id']], df['rating'])

Or if you have implicit data (0 or 1 for whether the user has interacted with 
the item), use model = Bernoulli(predictions)

Or if there are discrete scores (e.g. 1-10), then use a BetaBinomial 
