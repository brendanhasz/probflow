Latent Dirichlet Allocation
===========================

.. include:: ../macros.hrst

TODO: intro, link to colab w/ these examples

TODO: math


.. math::

    \begin{align}
    N_t & = \text{number of topics} \\
    N_w & = \text{number of words} \\
    N_d & = \text{number of documents} \\
    \boldsymbol{\varphi}_{k=1...N_t} & \sim \text{Dirichlet}_{N_w} (\boldsymbol{\beta}) \\
    \boldsymbol{\theta}_{d=1...N_d} & \sim \text{Dirichlet}_{N_t} (\boldsymbol{\alpha}) \\
    \mathbf{W} & \sim \text{OneHotCategorical}(\boldsymbol{\theta} \boldsymbol{\varphi})
    \end{align}


TODO: diagram

TODO: explain in terms of :math:`\boldsymbol{\varphi}` and :math:`\boldsymbol{\theta}` parameter matrixes

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python3

            import probflow as pf

            class LDA(pf.Model):

                def __init__(self, Nt, Nd, Nw):
                    self.phi = pf.DirichletParameter(Nw, Nt)   #per-topic word dists
                    self.theta = pf.DirichletParameter(Nt, Nd) #per-document topic dists

                def __call__(self, x):
                    probs = self.theta[x[:, 0]] @ self.phi()
                    return pf.OneHotCategorical(probs=probs)

    .. group-tab:: PyTorch

        .. code-block:: python3

            import probflow as pf
            import torch

            class LDA(pf.Model):

                def __init__(self, Nt, Nd, Nw):
                    self.phi = pf.DirichletParameter(Nw, Nt)   #per-topic word dists
                    self.theta = pf.DirichletParameter(Nt, Nd) #per-document topic dists

                def __call__(self, x):
                    x = torch.tensor(x)
                    probs = self.theta[x[:, 0]] @ self.phi()
                    return pf.OneHotCategorical(probs=probs)


To fit the model in this way, ``x`` will be document IDs, and ``y`` will be
a matrix of size ``(Ndocuments, Nwords)``.


.. code-block:: python3

    # Nt = number of topics to use
    # Nd = number of documents
    # Nw = number of words in the vocabulary
    # W = (Nd, Nw)-size matrix of per-document word probabilities
    doc_id = np.arange(W.shape[0])

    model = LDA(Nt, Nd, Nw)
    model.fit(doc_id, W)


TODO: Alternatively, when you have a LOT of documents, it's inefficient to
try and infer that huge Nd-by-Nt matrix of parameters, so you can use a neural
net to estimate the topic distribution from the word distributions (amortize).  Its...
kinda sorta like an autoencoder, where you're encoding documents into weighted
mixtures of topics, and then decoding the word distributions from those topic
distributions.

.. tabs::

    .. group-tab:: TensorFlow

        .. code-block:: python3

            class LdaNet(pf.Model):

                def __init__(self, dims):
                    self.phi = pf.DirichletParameter(dims[0], dims[-1])
                    self.net = pf.DenseNetwork(dims)

                def __call__(self, x):
                    probs = self.net(x) @ self.phi()
                    return pf.OneHotCategorical(probs=probs)

    .. group-tab:: PyTorch

        .. code-block:: python3

            class LdaNet(pf.Model):

                def __init__(self, dims):
                    self.phi = pf.DirichletParameter(dims[0], dims[-1])
                    self.net = pf.DenseNetwork(dims)

                def __call__(self, x):
                    x = torch.tensor(x)
                    probs = self.net(x) @ self.phi()
                    return pf.OneHotCategorical(probs=probs)


TODO: And then when fitting the model we'll use the per-document word frequency
matrix as both ``x`` and ``y``:


.. code-block:: python3

    model = LdaNet([Nw, 128, 128, 128, Nt])
    model.fit(W, W)
