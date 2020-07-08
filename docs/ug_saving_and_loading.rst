.. _ug_saving_and_loading:

Saving and Loading Models
=========================

.. include:: macros.hrst

Models and modules can be serialized and saved to file.  The easiest way is 
to use the :meth:`save <probflow.modules.Module.save>` method of |Model| 
and |Module| 

.. code-block:: python3

    # ...
    # model.fit(...)

    model.save('my_model.pfm')

    # Or, use pf.save, which does the same thing
    # pf.save(model, 'my_model')

Then the file can be loaded with :func:`pf.load <probflow.utils.io.load>`:

.. code-block:: python3

    model = pf.load('my_model.pfm')
    # model.is a pf.Model object

.. admonition:: Saving and loading only works between identical Python versions

    Currently, you can only load a ProbFlow model or module file using the 
    exact same version of Python as was used to save it.  This is because
    ProbFlow uses `cloudpickle <http://github.com/cloudpipe/cloudpickle>`_
    to serialize objects.  Fancier storage might be supported in the future
    but don't hold your breath!

Models and modules can also be serialized into byte strings.  To do this, use
the :meth:`dumps <probflow.modules.Module.dumps>` method of |Model| 
and |Module| 

.. code-block:: python3

    model_bytes = model.dumps()

    # Or, use pf.dumps, which does the same thing
    # model_bytes = pf.dumps(model)

Then the file can be loaded back from the byte string with 
:func:`pf.loads <probflow.utils.io.loads>`:

.. code-block:: python3

    model = pf.loads(model_bytes)
    # model.is a pf.Model object

