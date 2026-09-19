Selecting a Backend, Datatype, and Device
=========================================

.. include:: ../macros.hrst

Before building models with ProbFlow, you'll want to decide which backend to
use, and what default datatype to use.


Setting the Backend
-------------------

What I mean by "backend" is the system which performs the automatic
differentiation required to fit models with stochastic variational inference.
ProbFlow currently supports three backends: |TensorFlow|, |PyTorch|, and |JAX|.

ProbFlow detects which of the three you have installed and automatically sets
it as the backend.

However you can manually set which backend to use:

.. code-block:: python3

   import probflow as pf

   pf.set_backend(pf.ProbflowBackend.PYTORCH)  # or TENSORFLOW or JAX

You can see which backend is currently being used by:

.. code-block:: python3

   pf.get_backend()

ProbFlow will only use operations specific to the backend you've chosen, and
you can only use operations from your chosen backend when specifying your
models via ProbFlow.


Setting the Datatype
--------------------

You can also set the default datatype ProbFlow uses for creating the variable
tensors.  This datatype much match the datatype of the data you're fitting.
The default datatype is ``tf.dtypes.float32`` when TensorFlow is the backend,
``torch.float32`` when PyTorch is the backend, and ``jnp.float32`` when JAX is
the backend.

You can see which is the current default datatype with:

.. code-block:: python3

   pf.get_datatype()

And you can set the default datatype with ``pf.set_datatype``.  For example,
to instead use double precision with the TensorFlow backend:

.. code-block:: python3

   import tensorflow as tf
   
   pf.set_datatype(tf.dtypes.float64)

.. admonition:: Personal opinion warning!

   I'd gently recommend sticking to the default float32 datatype.  Variational
   inference is super noisy as is, so do we *really* need all that extra
   precision?  Single precision is also a lot faster on most GPUs.  If your
   data is of a different type, just cast it with (for numpy arrays and pandas
   DataFrames) ``.astype('float32')``.


.. _selecting-a-device:

Selecting a Device (CPU or GPU)
-------------------------------

You can select which device the ProbFlow backend should use for computations
(i.e., whether to use the CPU or GPU).

To see what the current device is, you can use:

.. code-block:: python3

   import probflow as pf

   pf.get_default_device()

ProbFlow does not provide tools for controlling the device selection, instead
you should use device management tools provided by the backend you are using:

.. tabs::

   .. group-tab:: TensorFlow

      TensorFlow automatically detects and uses a GPU if available. To force it
      to use the CPU even if a GPU is available, you can set the ``CUDA_VISIBLE_DEVICES``
      environment variable, **before** importing TensorFlow or ProbFlow:

      .. code-block:: python3

         import os
         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

         # NOTE: set the above env var BEFORE importing TensorFlow or ProbFlow
         import tensorflow as tf
         import probflow as pf


   .. group-tab:: PyTorch

      PyTorch does **NOT** automatically detect/use a GPU when available.  Even if
      your hardware has a GPU device, PyTorch will not use it unless you explicitly set the device.
      To use a GPU with PyTorch, explicitly set the device to 'cuda', **before**
      initializing any of your ProbFlow code.

      .. code-block:: python3

         import torch
         torch.set_default_device('cuda')

         # NOTE: set the above env var BEFORE importing ProbFlow
         # (technically just before initializing any ProbFlow models/modules)
         import probflow as pf

   .. group-tab:: JAX

      JAX automatically detects and uses a GPU if available. To force it to use the CPU
      even if a GPU is available, you can set the ``JAX_PLATFORMS`` environment
      variable, **before** importing JAX or ProbFlow:

      .. code-block:: python3

         import os
         os.environ["JAX_PLATFORMS"] = "cpu"

         # NOTE: set the above env var BEFORE importing JAX + ProbFlow
         import jax
         import jax.numpy as jnp
         import probflow as pf