Planned Improvements
====================

This page has a list of planned improvements, in order of when I plan to get to them.

Backlog (short term):
---------------------

* Finish BaseLayer, BaseModel.fit, Variable, and Input
* Tests and debug a simple 1d linear model
* Docs for BaseLayer, BaseModel.fit, Variable, and Input
* Tests which cover distributions, layers, and core elements that have been written (ensure right shapes, can take any valid combo of types as args, etc)
* Docs for distributions + layers
* Finish BaseModel methods
* Tests for BaseModel methods
* Docs for BaseModel methods
* Dense layer (w/ flipout)
* Test for Dense, and compare to stan or edward
* Sequential layer
* Tests
* Models which only use Dense
* Tests
* Bernoulli and Poisson dists
* Models which use them (Classifiers)
* Tests

Backlog (long term):
--------------------

* `Tensorflow graph view`_
* `Tensorflow dashboard`_
* `Embedding layer`_
* Neural Matrix Factorization
* Conv layers
* Pooling layers
* Ready-made Conv models
* `Model comparison`_
* `Support for random effects and multilevel models`_
* `Mixture distribution`_
* LSTM Layer



Notes
-----

Embedding layer
^^^^^^^^^^^^^^^

With priors on the embedding vectors to regularize.  

Tensorflow graph view
^^^^^^^^^^^^^^^^^^^^^

Should be able to show the tensorflow graph for a model.
Maybe via a something like ``model.tensorboard_graph(...same args as fit?...)``.
See https://www.tensorflow.org/guide/graph_viz


Tensorflow dashboard
^^^^^^^^^^^^^^^^^^^^

The ``fit()`` func should have a ``show_dashboard`` kwarg or something.  If true, 
opens the tensorboard while training.


Support for random effects and multilevel models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Should allow for random effects, mixed effects (just the results of a fixed effects net plus the results of a random effects net) and also hierarchical/multilevel models (where random effect variables are nested).
Ie for random effects there's an over all dist of weights, but each subject/group has their own weight distributions which are drawn from pop dist
Use the reparam trick?
And should be able to make multilevel model with that: eg individuals drawn from schools (in fact comparing to the 8 schools example in r would be good way to test that it works)
Perhaps make a RandomVariable() which takes a slice of the x_values placeholder? (as individual/group id or whatever)


Model comparison
^^^^^^^^^^^^^^^^

somehow.  AIC/BIC/DIC/WAIC/LOO?
I mean.  Or just use held-out log posterior prob...
or cross-validated summed log posterior prob?


Mixture distribution
^^^^^^^^^^^^^^^^^^^^

A continuous distribution which takes a list of other distrbutions.