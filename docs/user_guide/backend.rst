Selecting a Backend and Datatype
================================

.. include:: ../macros.hrst

Before building models with ProbFlow, you'll want to decide which backend to
use, and what default datatype to use.


Setting the Backend
-------------------

What I mean by "backend" is the system which performs the automatic
differentiation required to fit models with stochastic variational inference.
ProbFlow currently supports two backends: |TensorFlow| and |PyTorch|.
TensorFlow is the default backend, but you can set which backend to use:

.. code-block:: python3

   import probflow as pf

   pf.set_backend('pytorch') #or 'tensorflow'

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
and ``torch.float32`` when PyTorch is the backend.

You can see which is the current default datatype with:

.. code-block:: python3

   pf.get_datatype()

And you can set the default datatype like this.  For example, to instead use
double precision with the TensorFlow backend:

.. code-block:: python3

   pf.set_datatype(tf.dtypes.float64)

.. admonition:: Personal opinion warning!

   I'd gently recommend sticking to the default float32 datatype.  Variational
   inference is super noisy as is, so do we *really* need all that extra
   precision?  Single precision is also a lot faster on most GPUs.  If your
   data is of a different type, just cast it with (for numpy arrays and pandas
   DataFrames) ``.astype('float32')``.
