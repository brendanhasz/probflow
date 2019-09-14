.. _example_batch_normalization:

Batch Normalization
===================

.. include:: macros.hrst


TODO: intro, math, diagram


Batch normalization can be performed using the :class:`.BatchNormalization` 
Module.  For example, to add batch normalization to the dense neural network
from the :ref:`previous example <example_fully_connected>`:

.. code-block:: python3

    class DenseRegression(pf.Model):
        
        def __init__(self):
            self.net = pf.Sequential([
                pf.Dense(5, 128),
                pf.BatchNormalization(128),
                tf.nn.relu,
                pf.Dense(128, 64),
                pf.BatchNormalization(64),
                tf.nn.relu,
                pf.Dense(64, 1),
            ])
            self.s = pf.ScaleParameter()

        def __call__(self, x):
            return pf.Normal(self.net(x), self.s())

