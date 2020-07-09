.. _dev_guide:

Developer Guide
===============

.. include:: macros.hrst

At some point I'll fill this in a bit more, but for the time being:


Requirements
------------

First make sure you've got the following installed:

* make
* git
* python3
* `virtualenv <http://docs.python.org/3/library/venv.html>`_


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

Then create a virtual environment called ``venv`` for testing by running:

.. code-block:: bash

   python3 -m venv venv

To set up a development environment (this activates the virtual environment you
just created, installs the requirements, dev requirements, and the ProbFlow
package in editable mode from your version of the source code - see the
``Makefile`` for the commands it's running), run:

.. code-block:: bash

   make dev-env


Tests
-----

Then you can edit the source code, which is in ``src/probflow``.  The tests are
in ``tests``.  To run the tests, run

.. code-block:: bash

   make tests

If you get an error during the tests and want to debug, the tests are written
using `pytest <http://docs.pytest.org>`_, so to drop into the
`python debugger <http://docs.python.org/3/library/pdb.html>`_ on errors, run:

.. code-block:: bash

   . venv/bin/activate
   pytest tests/test_you_want_to_run.py --pdb


Style
-----

To run the autoformatting (using ``black``) and style checks (using
``flake8``), run

.. code-block:: bash

   make format


Documentation
-------------

To build the documentation locally (this creates html files in the
``docs/_html`` directory, the main page being ``docs/_html/index.html``), run:

.. code-block:: bash

   make documentation


Contributing your changes
-------------------------

Then if you want to contribute your changes, make a
`pull request <https://github.com/brendanhasz/probflow/pulls>`_!
