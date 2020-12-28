.. _dev_guide:

Developer Guide
===============

.. include:: ../macros.hrst

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

Then, to set up a development environment with tensorflow, run

.. code-block:: bash

   make init-tensorflow

or alternatively to set up a dev environment with pytorch,

.. code-block:: bash

   make init-tensorflow

The above command creates a new virtual environment called ``venv``, activates
that virtual environment, installs the requirements (including tensorflow or
pytorch), dev requirements, and the ProbFlow package in editable mode from your
version of the source code - see the ``Makefile`` for the commands it's
running).


Tests
-----

Then you can edit the source code, which is in ``src/probflow``.  The tests are
in ``tests``.  To run the tensorflow tests, run

.. code-block:: bash

   make test-tensorflow

and to run the PyTorch tests, run

.. code-block:: bash

   make test-pytorch

If you get an error during the tests and want to debug, the tests are written
using `pytest <http://docs.pytest.org>`_, so to drop into the
`python debugger <http://docs.python.org/3/library/pdb.html>`_ on errors, run:

.. code-block:: bash

   . venv/bin/activate
   pytest tests/test_you_want_to_run.py --pdb


Style
-----

To run the autoformatting (using ``isort`` and ``black``) and style checks
(using ``flake8``), run

.. code-block:: bash

   make format


Documentation
-------------

To build the documentation locally (the docs are written for and built with
`Sphinx <http://www.sphinx-doc.org>`_, this command creates html files in the
``docs/_html`` directory, the main page being ``docs/_html/index.html``), run:

.. code-block:: bash

   make docs


Contributing your changes
-------------------------

Then if you want to contribute your changes, make a
`pull request <https://github.com/brendanhasz/probflow/pulls>`_!
