bayesian_keras
==============

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   user-guide
   api

TODO: introduction, getting started (a few simple examples), installation, etc

.. _some-label-name:

Section
-------

*Italic* text

Subsection
^^^^^^^^^^

**Bold** text

Subsubsection
"""""""""""""

``Code sample``

* Bulleted
* list

  * nested
  * list

* con't

1. Numbered
2. list

This should be a hyperlink to `Keras <https://keras.io/>`_

.. You shouldn't be able to see me!

..
   Also shouldn't
   be able

   to see me

This is an :math:`\alpha=3` inline alpha

And here is a displayed math block:

.. math::

  y \sim \mathcal{N}(0, 1)

And a labeled equation:

.. math:: \beta \sim \text{Poisson}(\lambda=5)
   :label: beta_prior

The prior on :math:`\beta` is a Poisson distribution with rate parameter of 5 (see :eq:`beta_prior`).

Displayed code block. ::

  # And here's some code
  for i in range(5):
    print(i)

  # code block keeps going until un-indent
  
Normal text again

This should link to section :ref:`some-label-name`

This should link to the api page :doc:`/api`

This should have a custom link text :doc:`Custom link text </api>`