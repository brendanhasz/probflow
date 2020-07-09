.. _dev_guide:

Developer Guide
===============

.. include:: macros.hrst

At some point I'll fill this in a bit more, but for the time being: to start up
an environment to run and test ProbFlow, first make sure you've got the
following installed:

* ``make``
* ``python3``

If you don't already have virtualenv installed, run:

.. code-block:: bash

   python3 -m pip install --user virtualenv

Then, create a virtual environment called ``venv`` for testing:

.. code-block:: bash

   python3 -m venv venv

To set up a development environment, run (this activates the virtual
environment you just created, and installs the requirements, dev requirements,
and the ProbFlow package in editable mode from your version of the source code
- see the ``Makefile`` for the commands it's running):

.. code-block:: bash

   make dev-env

Then you can edit the source code, which is in ``src/probflow``.  The tests are
in ``tests``.  To run the tests, run

.. code-block:: bash

   make tests

To run the autoformatting (using ``black``) and style checks (using
``flake8``), run

.. code-block:: bash

   make format
