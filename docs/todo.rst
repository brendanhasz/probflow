Backlog
=======

This page has a list of planned improvements, in order of when I plan to get
to them.  If you're interested in tackling one of them, I'd be thrilled! 
`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!


Backlog
-------

* Speed tests on large dataset (tho keep in mind we're not using autograph for now, maybe just compare to w/o using autograph in raw TF code)
* Model evaluation methods (ones to be used in readme)
* Make model evaluation methods (predictive_sample, etc) handle generative models (x or y is None)
* Tests for those
* README / index
* User guide
* Examples
* Docs for everything implemented so far
* Fix issues so far (below)
* Merge back to main repo and release 2.0.0
* Different plotting methods for different types of dists (both for Parameter priors/posteriors and predictive distribution plots)
* All model evaluation methods + specialized types of models
* Make Module.trainable_variables return tf.Variables which are properties of module+sub-modules as well (and not neccesarily in parameters, also allow embedding of tf.Modules?)
* Real-world examples w/ public BigQuery datasets
* More examples: Hidden Markov models, Gaussian processes, and LDA
* Bayes estimate / decision methods
* Convolutional modules


Issues
------

* Autograph failure warnings? "Entity <blah blah> could not be transformed and will be executed as-is", where blah is "function sum" or "ound method Gamma.call of <probflow.distributions.Gamma..." etc.  Don't get the warnings when you remove the ``@tf.function`` in front of the ``train_step`` func defined in models.Model._train_step_tensorflow.  Happens both w/ CPU and GPU versions of TF 2.0
* LogisticRegression doesn't work at all! And seems to take a suspiciously long time...
* Model.metric (mae) causes too much memory usage (out of mem on colab w/ 100k sample linear regression?). Accidentally making a N^2 matrix maybe?
* Poisson currently requires y values to be floats? I think that's a TFP/TF 2.0 issue though (in their sc there's the line ``tf.maximum(y, 0.)``, which throws an error when y is of an int type).  Could cast inputs to float in pf.distributions.Poisson.__init__...
* Gamma distribution isn't passing the fit test (in tests/stats/test_distribution_fits)
* PyTorch support
* Add type hinting and enforcing

