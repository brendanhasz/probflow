Planned Improvements
====================

This page has a list of planned improvements, in order of when I plan to get to them.


Backlog (short term):
---------------------

* Let plot_posterior_over_training() plot prob distribution when prob=True, and tests
* Add support for Input w/ cols= a string (and input=pandas df)
* Finish BaseDistribution.fit()
* BaseDistribution.plot_predictive_distribution()
* `Better parameter initialization`_
* Tests and debug for validation split and shuffles
* Finish BaseLayer, BaseDistribution.fit, Parameter, and Input
* Docs for BaseLayer, BaseDistribution.fit, Parameter, and Input
* Tests which cover distributions, layers, and core elements that have been written (ensure right shapes, can take any valid combo of types as args, etc)
* Docs for distributions + layers
* Finish BaseDistribution critisism methods
* Tests for BaseDistribution critisism methods
* Docs for BaseDistribution critisism methods
* Dense layer (w/ flipout)
* Test for Dense, add Dense test to stats/test_LinearRegression
* `Sequential layer`_
* Tests
* Models which only use Dense
* Tests
* Bernoulli and Poisson dists
* `Mean alias for discrete dists`_
* Models which use them (Classifiers)
* Tests
* Write the docs main page and user guide
* Examples dir (with examples therein)
* `Reset method`_
* `Sklearn support`_

Backlog (long term):
--------------------

* `Tensorflow graph view`_
* `Tensorflow dashboard`_
* `Slicing`_
* `Embedding layer`_
* Neural Matrix Factorization
* Multivariate Normal, StudentT, and Cauchy dists
* Bayesian correlation example and Model
* `Separate model from noise uncertainty`_ 
* `Saving and loading and initializing parameters`_
* `Transfer learning`_
* `Bijector support`_? e.g so you can do ``model=Exp(Normal()); model.fit()``
* `Input data as tf dataset iterators`_
* `Model comparison`_
* `Support for random effects and multilevel models`_
* `Mixture distribution`_
* Conv layers
* Pooling layers
* Ready-made Conv models
* LSTM Layer


Notes
-----

Better parameter initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scale param should initialize w/ mean of 1 and SD of... Something reasonable.
Also make sure the initializer your're using for parameter values is reasonable
And make the STD devs variables also initialize to 1


Sequential layer
^^^^^^^^^^^^^^^^

Sequential layer can't be a class which inherits from BaseLayer b/c it takes a list.  Also, elements of that list will be instantiated Layers.  Will have to be a func which sets the arg['input'] of each sucessive element as the output of the last layer and then return the last layer?

Also each non-terminal layer in a Sequential layer's list can only have 1 output (and each non-first layer can only have 1 input).


Mean alias for discrete dists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Categorical distributions should have a mean function (for convenience) which actually returns the mode (also have a mode func). That way if you call predict on a model which involves a categorical dist it'll work just fine (by recursively evaluating mean())


Reset method
^^^^^^^^^^^^

Models should have a reset() method which sets is_fit to false and clears the tf graph. Then, in fit, only builds the model if is_fit is false. That way you can do transfer learning or snapshot ensembling easily: fit to one set of data, then fit to another, and for the second fit the parameters start where they were at the end of the first fit. But if you want to explicitly re fit from scratch call model.reset()
Ideally calling reset on a model would *only* reset the variables contained in that model, and not the entire TF graph...

It should also close the tf session.


Sklearn support
^^^^^^^^^^^^^^^

Model classes should be consistent with a sklearn estimator. 
Or if that won't work, include a sklearn Estimator which takes a model obj.
https://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator


Slicing
^^^^^^^

NOTE that you've implemented this, just need to test/debug it.
Added layers.Gather and Parameter.__getitem__ which uses Gather.

Ability to 'slice' arrays, e.g.:

.. code-block:: python

   inds = Input()
   values = Variable(shape[n_unique_inds,1])
   values[inds]

This will enable the user to do embeddings,

.. code-block:: python

   user_ids = Input('user ids')
   item_ids = Input('user ids')
   user_embeddings = Parameter(shape=[n_users, 50])
   item_embeddings = Parameter(shape=[n_items, 50])
   predictions = Dot(user_embeddings[user_ids],
                     item_embeddings[item_ids])

mixed effects,

.. code-block:: python

  subj_id = Input('subject')
  mixed_eff = Parameter(shape=n_subj)
  predictions = mixed_eff[subj_id]

and multilevel models:

.. code-block:: python

  pop_mean = Parameter()
  pop_std = ScaleParameter()
  subj_params = Parameter(shape=n_subj,
                          prior=Normal(pop_mean, pop_std))
  subj_id = Input('subject')
  params = subj_params[subj_id]

using tf.gather() under the hood.  
how does np implement that?  Ok looks like via __getitem__
which should be added to Parameter (can't slice on layers)
see https://docs.python.org/3/reference/datamodel.html#object.__getitem__


Tensorflow graph view
^^^^^^^^^^^^^^^^^^^^^

Should be able to show the tensorflow graph for a model.
Maybe via a something like ``model.tensorboard_graph(...same args as fit?...)``.
See https://www.tensorflow.org/guide/graph_viz

Also should handle scoping better so the tensorboard graph view of models isn't
so hideous...

Save graph w/ 

.. code-block:: python

   writer = tf.summary.FileWriter("path\to\log", sess.graph)

and remember to do ``writer.close()`` at some point.


Tensorflow dashboard
^^^^^^^^^^^^^^^^^^^^

The ``fit()`` func should have a ``show_dashboard`` kwarg or something.  If true, 
opens the tensorboard while training.

Set up the TF stuff in python (see previous section).

Then start tensorboard.  May have to use subprocess.Popen (part of std lib):

.. code-block:: python

   import subprocess
   subprocess.Popen(['tensorboard' '--logdir=path\to\log'])

And finally open a web browser to the tensorboard w/ the webbrowser package (also part of std lib)

.. code-block:: python

   import webbrowser
   webbrowser.open('localhost:6006', new=2)


Embedding layer
^^^^^^^^^^^^^^^

With priors on the embedding vectors to regularize.  


Separate model from noise uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Right now predictive_distribution estimates the total uncertainty. Would be nice to be able to separately estimate model uncertainty (aka epistemic unc) vs noise uncertainty (aka aleatoric unc).  Could estimate just the model uncertainty by taking the mean if the sample model? Ie _built_model.mean()


Saving and loading and initializing parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Should have a way to save and load models, layers, parameters (and their posterior variable values!).  h5?  Or just pickle even?

Also should be able to initialize parameter posterior variables to a specific value (a feature which would probably be used when loading a model/parameter).


Transfer learning
^^^^^^^^^^^^^^^^^

Ideally, you can train a model, then take the parameters or even whole layers (with trees of parameters and layers within them) from that trained model, and plug it into a new model and train that new model.

Also, should be able to set whether parameters are trainable. Or layers (which just sets the trainable value of all parameters contained in that layer or its children).
E.g. for transfer learning, you might want to train a model, take some layer(s) from it, add a few layers on top, and then train *only those new layers* you added on top, so you'd want to set trainable=False for the layer(s) which were pre-trained.

Could go through the tree and for all parameters set their posterior parameter 
tf.Varable's .trainable property = False?


Bijector support
^^^^^^^^^^^^^^^^

Adding the jacobian adjustment isn't too bad, just add Abs( d transform / dt ).
But you also then need to worry about doing the *inverse* transform.
E.g. w/ ``y ~ Exp(Normal(mu, sigma))``, Exp layer needs to *inverse* transform y
(i.e. take ``ln(y)``), compute prob of ``ln(y) ~ N(mu, sigma)``, and then 
return that prob plus the Jacobian adjustment.

But, don't need a special "bijector" or anything, just add that functionality
to the Exp layer (and other transform layers, like Reciprocal, Log, and Sigmoid)

Also, is there a way to get mean() to work w/ Bijectors? TFP currently just throws an error when you try to call mean on a bijected dist. Currently mean() won't return the mean for transformed dists b/c for example mean(exp(x)) isn't the same as exp(mean(x)).  I don't think getting that to work is as easy as it is for the log prob (were you just transform or inv transform the values), because there's no principled way to get the mean of a transformed dist, and some transforms don't even have analytically tractable means (e.g. the logit normal dist).


Input data as tf dataset iterators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The big advantage of bayes by backprop w/ tensorflow is your data doesn't have
to fit into memory.  Right now, ``BaseDistribution.fit`` assumes its inputs
``x`` and ``y`` are numpy arrays (or pandas arrays).  
Though I guess you could use memory mapping if it won't fit in memory.
Distributed arrays would be hard though.  Dask maybe?
Anyway, it would be nice 
to let it take dataset iterators so users can define their own data pipelines.


Support for random effects and multilevel models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Should allow for random effects, mixed effects (just the results of a fixed effects net plus the results of a random effects net) and also hierarchical/multilevel models (where random effect variables are nested).
Ie for random effects there's an over all dist of weights, but each subject/group has their own weight distributions which are drawn from pop dist
Use the reparam trick?
And should be able to make multilevel model with that: eg individuals drawn from schools (in fact comparing to the 8 schools example in r would be good way to test that it works)
Perhaps make a RandomVariable() which takes a slice of the x_values placeholder? (as individual/group id or whatever)


Model comparison
^^^^^^^^^^^^^^^^

AIC/BIC/DIC/WAIC/LOO?
I mean.  Or just use held-out log posterior prob...
or cross-validated summed log posterior prob?


Mixture distribution
^^^^^^^^^^^^^^^^^^^^

A continuous distribution which takes a list of other distrbutions.
