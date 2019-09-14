.. _ug_backend:

Selecting a Backend
===================

.. include:: macros.hrst

Before building models with ProbFlow, you'll want to decide which backend
to use.  ProbFlow currently supports two backends: |TensorFlow| and |PyTorch|.
TensorFlow is the default, but you can set which backend to use:

.. code-block:: python3

	import probflow as pf

	pf.set_backend('pytorch') #or 'tensorflow'

You can see which backend is currently being used by:

.. code-block:: python3

	pf.get_backend()

ProbFlow will only use operations specific to the backend you've chosen,
and you can only use operations from your chosen backend when specifying your
models via ProbFlow.
