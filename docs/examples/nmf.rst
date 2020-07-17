Neural Matrix Factorization
===========================

.. include:: ../macros.hrst


TODO: description...

Matrix Factorization
--------------------

TODO: for a vanilla matrix factorization, description, diagram, math
(with binary interactions)

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python3

            import probflow as pf
            import tensorflow as tf

            class MatrixFactorization(pf.Model):

                def __init__(self, Nu, Ni, Nd):
                    self.user_emb = pf.Embedding(Nu, Nd)
                    self.item_emb = pf.Embedding(Ni, Nd)

                def __call__(self, x):
                    user_vec = self.user_emb(x['user_id'])
                    item_vec = self.item_emb(x['item_id'])
                    logits = user_vec @ tf.transpose(item_vec)
                    return pf.Bernoulli(logits)

    .. group-tab:: PyTorch

        .. code-block:: python3

            import probflow as pf
            import torch

            class MatrixFactorization(pf.Model):

                def __init__(self, Nu, Ni, Nd):
                    self.user_emb = pf.Embedding(Nu, Nd)
                    self.item_emb = pf.Embedding(Ni, Nd)

                def __call__(self, x):
                    user_vec = self.user_emb(torch.tensor(x['user_id']))
                    item_vec = self.item_emb(torch.tensor(x['item_id']))
                    logits = user_vec @ torch.t(item_vec)
                    return pf.Bernoulli(logits)


TODO: Then can instantiate the model

.. code-block:: python3

    #df = DataFrame w/ 3 columns: 'user_id', 'item_id', and 'rating'
    Nu = df['user_id'].nunique() #number of users
    Ni = df['item_id'].nunique() #number of items
    Nd = 50 #number of embedding dimensions
    model = MatrixFactorization(Nu, Ni, Nd)

TODO: Then fit it;

.. code-block:: python3

    model.fit(df[['user_id', 'item_id']], df['rating'])


Neural Collaborative Filtering
------------------------------

TODO: description, diagram, math
TODO: cite https://arxiv.org/abs/1708.05031

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python3

            class MatrixFactorization(pf.Model):

                def __init__(self, Nu, Ni, Nd, dims):
                    self.user_emb = pf.Embedding(Nu, Nd)
                    self.item_emb = pf.Embedding(Ni, Nd)
                    self.net = pf.DenseNetwork(dims)

                def __call__(self, x):
                    user_vec = self.user_emb(x['user_id'])
                    item_vec = self.item_emb(x['item_id'])
                    logits = self.net(tf.concat([user_vec, item_vec], axis=1))
                    return pf.Bernoulli(logits)

    .. group-tab:: PyTorch

        .. code-block:: python3

            class MatrixFactorization(pf.Model):

                def __init__(self, Nu, Ni, Nd, dims):
                    self.user_emb = pf.Embedding(Nu, Nd)
                    self.item_emb = pf.Embedding(Ni, Nd)
                    self.net = pf.DenseNetwork(dims)

                def __call__(self, x):
                    user_vec = self.user_emb(torch.tensor(x['user_id']))
                    item_vec = self.item_emb(torch.tensor(x['item_id']))
                    logits = self.net(torch.cat([user_vec, item_vec], 1))
                    return pf.Bernoulli(logits)


Neural Matrix Factorization
---------------------------

or for neural matrix factorization https://arxiv.org/abs/1708.05031

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python3

            class NeuralMatrixFactorization(pf.Model):

                def __init__(self, Nu, Ni, Nd, dims):
                    self.user_mf = pf.Embedding(Nu, Nd)
                    self.item_mf = pf.Embedding(Ni, Nd)
                    self.user_ncf = pf.Embedding(Nu, Nd)
                    self.item_ncf = pf.Embedding(Ni, Nd)
                    self.net = pf.DenseNetwork(dims)
                    self.linear = pf.Dense(dims[-1]+Nd)

                def __call__(self, x):
                    user_mf = self.user_mf(x['user_id'])
                    item_mf = self.item_mf(x['item_id'])
                    user_ncf = self.user_ncf(x['user_id'])
                    item_ncf = self.item_ncf(x['item_id'])
                    preds_mf = user_mf*item_mf
                    preds_ncf = self.net(tf.concat([user_ncf, item_ncf], axis=1))
                    logits = self.linear(tf.concat([preds_mf, preds_ncf], axis=1))
                    return pf.Bernoulli(logits)

    .. group-tab:: PyTorch

        .. code-block:: python3

            class NeuralMatrixFactorization(pf.Model):

                def __init__(self, Nu, Ni, Nd, dims):
                    self.user_mf = pf.Embedding(Nu, Nd)
                    self.item_mf = pf.Embedding(Ni, Nd)
                    self.user_ncf = pf.Embedding(Nu, Nd)
                    self.item_ncf = pf.Embedding(Ni, Nd)
                    self.net = pf.DenseNetwork(dims)
                    self.linear = pf.Dense(dims[-1]+Nd)

                def __call__(self, x):
                    uid = torch.tensor(x['user_id'])
                    iid = torch.tensor(x['item_id'])
                    user_mf = self.user_mf(uid)
                    item_mf = self.item_mf(iid)
                    user_ncf = self.user_ncf(uid)
                    item_ncf = self.item_ncf(iid)
                    preds_mf = user_mf*item_mf
                    preds_ncf = self.net(torch.cat([user_ncf, item_ncf], 1))
                    logits = self.linear(torch.cat([preds_mf, preds_ncf], 1))
                    return pf.Bernoulli(logits)


TODO: Then can instantiate the model

.. code-block:: python3

    #df = DataFrame w/ 3 columns: 'user_id', 'item_id', and 'rating'
    Nu = df['user_id'].nunique() #number of users
    Ni = df['item_id'].nunique() #number of items
    Nd = 50 #number of embedding dimensions
    dims = [Nd*2, 128, 64, 32]
    model = NeuralMatrixFactorization(Nu, Ni, Nd, dims)

TODO: Then fit it;

.. code-block:: python3

    model.fit(df[['user_id', 'item_id']], df['rating'])
