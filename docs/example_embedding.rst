.. _example_embedding:

Entity Embeddings
=================

.. include:: macros.hrst


TODO: explain embeddings, using the :class:`.Embedding` module

TODO: model, embedding of categorical features (1st column of x) and combining
with continuous features (rest of x) and predicting with fully-connected neural
net on top, which does binary classification

.. code-block:: python

    import probflow as pf

    class EmbeddingRegression(pf.Model):

        def __init__(self, k, Dcat, Dcon):
            self.emb = pf.Embedding(k, Dcat)
            self.net = pf.DenseNetwork([Dcat+Dcon, 1])

        def __call__(self, x):
            embeddings = self.emb(x[:, 0])
            logits = self.net(tf.concat([embeddings, x[:, 1:]], -1))
            return pf.Bernoulli(logits)
