.. _dev_guide:

Developer Guide
===============

.. include:: ../macros.hrst

At some point I'll fill this in a bit more, but for the time being:


Requirements
------------

First make sure you've got the following installed:

* git
* `uv <https://docs.astral.sh/uv/>`_
* make


Setting up a development environment
------------------------------------

To start up an environment to run and test ProbFlow, first make a fork of the
`ProbFlow github repository <https://github.com/brendanhasz/probflow>`_.
Then, clone your fork to download the repository to your machine (this assumes
you're connecting to github
`using ssh <https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh>`_
):

.. code-block:: bash

   git clone git@github.com:<your_github_username>/probflow.git
   cd probflow

Then, to set up a development environment with probflow and required packages installed, run

.. tabs::

    .. group-tab:: TensorFlow

         .. code-block:: bash

            make install

    .. group-tab:: PyTorch

         .. code-block:: bash

            make install BACKEND=pytorch

    .. group-tab:: JAX

         .. code-block:: bash

            make install BACKEND=jax

The above command creates a virtual environment (via ``uv``), and installs
the requirements (including tensorflow or pytorch or jax), dev requirements, and the
ProbFlow package in editable mode from your version of the source code - see
the ``Makefile`` for the commands it's running).


Tests
-----

Then you can edit the source code, which is in ``src/probflow``.  The tests are
in ``tests``.  To run all the tests, run:

.. code-block:: bash

   make test

The above command runs several types of tests including unit tests and statistical tests.

Unit tests are broken down into _shared_ tests (which can be run using either the PyTorch or TensorFlow backends),
and backend-specific tests.  The shared unit tests are in `tests/shared` and the backend-specific tests are in
`tests/<backend>`.  To run the unit tests just for a specific backend, run:

.. tabs::

    .. group-tab:: TensorFlow

         .. code-block:: bash

            make test-unit BACKEND=tensorflow

    .. group-tab:: PyTorch

         .. code-block:: bash

            make test-unit BACKEND=pytorch
            
    .. group-tab:: JAX

         .. code-block:: bash

            make test-unit BACKEND=jax

There are also statistical tests, which are in `tests/stats`, and which check that the models are accurately able to fit data.
These are all backend-independent tests, but can be run using either backend.  To run the statistical tests, run:

.. tabs::

    .. group-tab:: TensorFlow

         .. code-block:: bash

            make test-stats BACKEND=tensorflow

    .. group-tab:: PyTorch

         .. code-block:: bash

            make test-stats BACKEND=pytorch
            
    .. group-tab:: JAX

         .. code-block:: bash

            make test-stats BACKEND=jax

If you get an error during the tests and want to debug, the tests are written
using `pytest <http://docs.pytest.org>`_, so to drop into the
`python debugger <http://docs.python.org/3/library/pdb.html>`_ on errors, run:

.. code-block:: bash

   make install BACKEND=<desired backend>
   uv run pytest tests/test_you_want_to_run.py --pdb


Style
-----

To run the autoformatting, style checks (using ``ruff``), and typing checks (using ``mypy``), run

.. code-block:: bash

   make format


Version bumping
---------------

To automatically bump the minor or patch version number, run ``make bump-minor`` or ``make bump-patch``.
This will update the version number in ``pyproject.toml``.
You can then commit that change, push it to your fork, and make a pull request.


Documentation
-------------

To build the documentation locally (the docs are written for and built with
`Sphinx <http://www.sphinx-doc.org>`_, this command creates html files in the
``docs/_html`` directory, the main page being ``docs/_html/index.html``), run:

.. code-block:: bash

   make docs


Benchmarking
------------

To run the benchmarking for all backends and write the results to the documentation, run:

.. code-block:: bash

   make benchmark BENCHMARKING_DEVICE=<device>

Where ``<device>`` is the device your current hardware is using (``cpu`` or ``gpu``).
For more on that, see :ref:`selecting-a-device`.

This fits a linear regression on datasets of varying size and dimensionality, for each backend.


Contributing your changes
-------------------------

Then if you want to contribute your changes, make a
`pull request <https://github.com/brendanhasz/probflow/pulls>`_!
