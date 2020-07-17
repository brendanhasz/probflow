Saving and Loading Models
=========================

.. include:: ../macros.hrst

Models and modules can be saved to a file or serialized to JSON-safe strings.

Saving a model to disk
----------------------

The easiest way to save a ProbFlow |Model| (this will also work for a |Module|
object) is to use the :meth:`save <probflow.modules.Module.save>` method:

.. code-block:: python3

    # ...
    # model.fit(...)

    model.save('my_model.pfm')

    # Or, use pf.save, which does the same thing
    # pf.save(model, 'my_model')

Then the file can be loaded with :func:`pf.load <probflow.utils.io.load>`

.. code-block:: python3

    model = pf.load('my_model.pfm')
    # model is a pf.Model object

Note that the file extension doesn't matter - i.e. it doesn't have to be
``.pfm``, that's just for "ProbFlow Model" (to give it an air of legitimacy
haha).  You could just as easily use ``.pkl``, or ``.dat`` or whatever.

.. admonition:: Saving and loading only works between identical Python versions

    Currently, you can only load a ProbFlow model or module file using the
    exact same version of Python as was used to save it.  This is because
    ProbFlow uses `cloudpickle <http://github.com/cloudpipe/cloudpickle>`_ to
    serialize objects.  Fancier storage might be supported in the future but
    don't hold your breath!


Serializing a model to a JSON-safe string
-----------------------------------------

Models and modules can also be serialized into JSON-safe strings.  To do this,
use the :meth:`dumps <probflow.modules.Module.dumps>` method of |Model| and
|Module|

.. code-block:: python3

    model_str = model.dumps()

    # Or, use pf.dumps, which does the same thing
    # model_str = pf.dumps(model)

These are UTF-8 encoded strings, so they're JSON-safe. That is, you can do
this:

.. code-block:: python3

    import json
    json.dumps({"model": model.dumps()})

Then the file can be loaded back from the string with
:func:`pf.loads <probflow.utils.io.loads>`

.. code-block:: python3

    model = pf.loads(model_str)
    # model is a pf.Model object

