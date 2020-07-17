Utils
=====

.. include:: ../macros.hrst

The :mod:`.utils` module contains utility classes, functions, and settings
which ProbFlow uses internally.  The sub-modules of :mod:`.utils` are:

* :mod:`.settings` - backend, datatype, and sampling settings
* :mod:`.base` - abstract base classes for ProbFlow objects
* :mod:`.ops` - backend-independent mathematical operations
* :mod:`.casting` - backend-independent casting operations
* :mod:`.initializers` - backend-independent variable initializer functions
* :mod:`.io` - functions for loading and saving models
* :mod:`.metrics` - functions for computing various model performance metrics
* :mod:`.plotting` - functions for plotting distributions, posteriors, etc
* :mod:`.torch_distributions` - manual implementations of missing torch dists
* :mod:`.validation` - functions for data type validation

Settings
--------

.. automodule:: probflow.utils.settings
   :members:
   :inherited-members:
   :show-inheritance:


Base
----

.. automodule:: probflow.utils.base
   :members:
   :inherited-members:
   :show-inheritance:


Ops
---

.. automodule:: probflow.utils.ops
   :members:
   :inherited-members:
   :show-inheritance:


Casting
-------

.. automodule:: probflow.utils.casting
   :members:
   :inherited-members:
   :show-inheritance:


Initializers
------------

.. automodule:: probflow.utils.initializers
   :members:
   :inherited-members:
   :show-inheritance:


IO
--

.. automodule:: probflow.utils.io
   :members:
   :inherited-members:
   :show-inheritance:


Metrics
-------

.. automodule:: probflow.utils.metrics
   :members:
   :inherited-members:
   :show-inheritance:


Plotting
--------

.. automodule:: probflow.utils.plotting
   :members:
   :inherited-members:
   :show-inheritance:


Torch Distributions
-------------------

.. automodule:: probflow.utils.torch_distributions
   :members:
   :inherited-members:
   :show-inheritance:


Validation
----------

.. automodule:: probflow.utils.validation
   :members:
   :inherited-members:
   :show-inheritance:
