Fitting a Model
===============

.. include:: ../macros.hrst

TODO:

* basic example of fitting w/ numpy arrays
* fitting changing the batch_size, epochs, lr, and shuffle
* fitting w/ a custom optimizer and/or optimizer kwargs
* fitting w/ or w/o flipout
* passing a pandas dataframe
* passing a DataGenerator to fit

Using multiple MC samples per batch
-----------------------------------

By default, ProbFlow uses only one Monte Carlo sample from the variational
posteriors per batch.  However, you can use more by passing the `n_mc` keyword
argument to :meth:`.Model.fit`.  For example, to use 10 MC samples during
training:

.. code-block:: python3

    model = pf.LinearRegression(x.shape[1])

    model.fit(x, y, n_mc=10)

Using more MC samples will cause the fit to take longer, but the parameter
optimization will be much more stable because the variance of the gradients
will be less.

Note that :class:`.Dense` modules, which use the flipout estimator by default,
will not use flipout when `n_mc` > 1.


Backend graph optimization during fitting
-----------------------------------------

By default, ProbFlow uses
`tf.function <https://www.tensorflow.org/api_docs/python/tf/function>`_
(for TensorFlow) or
`tracing <https://pytorch.org/docs/master/generated/torch.jit.trace.html>`_
(for PyTorch) to optimize the gradient computations during training.  This
generally makes training faster.

.. code-block:: python3

    N = 1024
    D = 7
    randn = lambda *a: np.random.randn(*a).astype('float32')
    x = randn(N, D)
    w = randn(D, 1)
    y = x@w + 0.1*randn(N, 1)

    model = pf.LinearRegression(D)

    model.fit(x, y)
    # takes around 5s

But to disable autograph/tracing and use only eager execution during model
fitting, just pass the ``eager=True`` kwarg to ``fit``.  This takes longer but
can be more flexible in certiain situations that autograph/tracing can't
handle.

.. code-block:: python3

    model.fit(x, y, eager=True)
    # takes around 28s

.. warning::

    When inputs are |DataFrames| or |Series| it is not possible to use tracing
    or ``tf.function``, so ProbFlow falls back on eager execution by defualt
    when the input data are |DataFrames| or |Series|

It's much easier to debug models in eager mode, since you can step through your
own code using `pdb <https://docs.python.org/3/library/pdb.html>`_, instead of
trying to step through the tensorflow or pytorch compilation functions.  So, if
you're getting an error when fitting your model and want to debug the problem,
try using ``eager=True`` when calling ``fit``.

However, eager mode is used for all other ProbFlow functionality (e.g.
:meth:`.Model.predict`, :meth:`.Model.predictive_sample`,
:meth:`.Model.metric`, :meth:`.Model.posterior_sample`, etc).  If you want an
optimized version of one of ProbFlow's inference-time methods, for TensorFlow
you can wrap it in a ``tf.function``:

.. code-block:: python3

    #model.fit(...)

    @tf.function
    def fast_predict(X):
        return model.predict(X)

    fast_predict(x_test)

Or for PyTorch, use ``torch.jit.trace``:

.. code-block:: python3

    #model.fit(...)

    def predict_fn(X):
        return model.predict(X)

    fast_predict = torch.jit.trace(predict_fn, (example_x))

    fast_predict(x_test)

