.. _example_ar:

Autoregressive Models
=====================

.. include:: macros.hrst


TODO: description... 


AR(K) Model
-----------

TODO: math

TODO: diagram

.. code-block:: python3

    class AutoregressiveModel(pf.Model):

        def __init__(self, k):
            self.beta = pf.Parameter([k, 1])
            self.mu = pf.Parameter()
            self.sigma = pf.ScaleParameter()

        def __call__(self, x):
            preds = x @ self.beta() + self.mu()
            return pf.Normal(preds, self.sigma())

If we have a timeseries ``x``,

.. code-block:: python

    N = 1000
    x = np.linspace(0, 10, N) + np.random.randn(N)

Then we can pull out ``k``-length windows into a feature matrix ``X``, such that each row of ``X`` corresponds to a single time point for which we are trying to predict the next sample's value (in ``y``).

.. code-block:: python3

    k = 50

    X = np.empty([N-k-1, k])
    y = x[k:]
    for i in range(N-k-1):
        X[i, :] = x[i:i+k]

Then, we can create and fit the model:

.. code-block:: python3

    model = AutoregressiveModel(k)
    model.fit(X, y)

Note that this is exactly equivalent to just doing a linear regression on the lagged feature matrix:

.. code-block:: python3

    model = pf.LinearRegression(k)
    model.fit(X, y)



ARIMA Model
-----------

TODO: math

TODO: diagram

TODO: code
