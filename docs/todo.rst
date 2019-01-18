Planned Improvements
====================

This page has a list of planned improvements, in order of when I plan to get to them.

Backlog (short term):
---------------------

* Matrix multiply layer?
* Implement @ operator for layers (see PEP 465)
* Layers w/ multiple args should ensure their args are broadcastable
* fit should have a record=False arg (record posterior values over training)
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
* `Mean alias for discrete dists`_
* Models which use them (Classifiers)
* Tests
* `Reset method`_
* `Sklearn support`_

Backlog (long term):
--------------------

* `Tensorflow graph view`_
* `Tensorflow dashboard`_
* `Embedding layer`_
* Neural Matrix Factorization
* Multivariate Normal, StudentT, and Cauchy dists
* Bayesian correlation example and Model
* Conv layers
* Pooling layers
* Ready-made Conv models
* `Model comparison`_
* `Support for random effects and multilevel models`_
* `Mixture distribution`_
* LSTM Layer

Issues
------

Transformed mean does not return mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently mean() won't actually return the mean of a variable if that variable is transformed/bounded, bc mean(exp(x)) isn't the same as exp(mean(x)) ; currently you do the second one in Variable.mean(), and also in Exp(Variable()).
It's not exactly the end of the world: if the variatinional dist used is Normal, it'll return the mode.
But, it's technically not correct.


Notes
-----

Mean alias for discrete dists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Categorical distributions should have a mean function (for convenience) which actually returns the mode (also have a mode func). That way if you call predict on a model which involves a categorical dist it'll work just fine (by recursively evaluating mean())

Reset method
^^^^^^^^^^^^
Models should have a reset() method which sets is_fit to false and clears the tf graph. Then, in fit, only builds the model if is_fit is false. That way you can do transfer learning or snapshot ensembling easily: fit to one set of data, then fit to another, and for the second fit the parameters start where they were at the end of the first fit. But if you want to explicitly re fit from scratch call model.reset()
Ideally calling reset on a model would *only* reset the variables contained in that model, and not the entire TF graph...

Sklearn support
^^^^^^^^^^^^^^^

Model classes should be consistent with a sklearn estimator. 
Or if that won't work, include a sklearn Estimator which takes a model obj.
https://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator

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
