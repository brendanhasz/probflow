Backlog
=======

This page has a list of planned improvements, in order of when I plan to get
to them.  If you're interested in tackling one of them, I'd be thrilled! 
`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!


Backlog
-------

* User guide
* Examples
* Docs for everything implemented so far
* Fix issues so far (below)
* Release 2.0.0
* Add HiddenMarkovModel and GaussianProcess distributions + examples
* Add survival/churn modeling example + Exponential distribution
* Models should be able to return a backend distribution (not a PF distribution - do you even need PF distributions?).
* Finish pred_dist_plot for Discrete and Categorical Model, as well as calibration_curve for categorical.
* Make Parameters have an option for how to plot their posteriors/priors (ie use plot_dist, plot_discrete_dist, or plot_categorical_dist), and set defaults that make sense for parameters (e.g. default for Categorical param should be plot_categorical_dist)
* Add model saving/loading (Model.save_model and load_model)
* Summary method for Modules + Models which show hierarchy of modules/parameters
* Make Module.trainable_variables return tf.Variables which are properties of module+sub-modules as well (and not neccesarily in parameters, also allow embedding of tf.Modules?)
* Real-world examples w/ public BigQuery datasets
* Bayes estimate / decision methods
* Convolutional modules


Issues
------

* Autograph fails. "Entity <blah blah> could not be transformed and will be executed as-is", where blah is "function sum" or "bound method Gamma.call of <probflow.distributions.Gamma..." etc.  Don't get the warnings when you remove the ``@tf.function`` in front of the ``train_step`` func defined in models.Model._train_step_tensorflow.  Happens both w/ CPU and GPU versions of TF 2.0.  Presumably b/c autograph doesn't handle various python features (lambda funcs, fancy list comprehensions, methods, etc) used.  Runs about 3x slower w/o autograph optimization (just running w/ eager).
* LogisticRegression doesn't work at all! And seems to take a suspiciously long time...
* Model.metric (mae) causes too much memory usage (out of mem on colab w/ 100k sample linear regression?). Accidentally making a N^2 matrix maybe?
* Poisson currently requires y values to be floats? I think that's a TFP/TF 2.0 issue though (in their sc there's the line ``tf.maximum(y, 0.)``, which throws an error when y is of an int type).  Could cast inputs to float in pf.distributions.Poisson.__init__...
* Gamma distribution isn't passing the fit test (in tests/stats/test_distribution_fits) Fitting a Poisson GLM doesn't seem to be doing too well either...
* PyTorch support
* Add type hinting and enforcing
* Implement + test mean() for InverseGamma, Bernoulli, Categorical, and OneHotCategorical for pytorch
* Implement mixture distribution w/ pytorch backend. They're working on a MixtureSameFamily distribution for PyTorch (https://github.com/pytorch/pytorch/pull/22742) so maybe wait for that.
* Allow learning rate to be updated w/ PyTorch
