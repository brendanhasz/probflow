Backlog
=======

This page has a list of planned improvements, in order of when I plan to get
to them (the backlog).  If you're interested in tackling one of them, I'd be 
thrilled!  `Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!  There's also a list of things which I think are out of
scope for the package.


Backlog
-------

* User guide
* Examples
* Docs for everything implemented so far
* Fix some issues (below)
* Release 2.0.0
* Get autograph and pytorch working!
* Add HiddenMarkovModel and GaussianProcess distributions + examples
* Add survival/churn modeling example w/ Exponential or Gumbel distribution
* Finish pred_dist_plot for Discrete and Categorical Model, as well as calibration_curve for categorical.
* Make Parameters have an option for how to plot their posteriors/priors (ie use plot_dist, plot_discrete_dist, or plot_categorical_dist), and set defaults that make sense for parameters (e.g. default for Categorical param should be plot_categorical_dist)
* Add model saving/loading (Model.save_model and load_model)
* Summary method for Modules + Models which show hierarchy of modules/parameters
* Make Module.trainable_variables return tf.Variables which are properties of module+sub-modules as well (and not neccesarily in parameters, also allow embedding of tf.Modules?)
* Bayes estimate / decision methods
* Convolutional modules


Out of scope
------------

Things which would conceivably be cool, but I think are out of scope for the
package:

* Forms of inference aside from stochastic variational inference (e.g. MCMC, Stein methods, Langevin dynamics, Laplace approximation, importance sampling, expectation maximization, particle filters, finding purely the MAP/ML estimate, Bayesian estimation via dropout, etc...)
* Support for sequence models like RNNs and HMMs (though maybe someday will support that, but ProbFlow is aimed more towards supporting tabular/matrix data as opposed to sequence data.  Note that you could probably hack together a HMM using `tfd.HiddenMarkovModel <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel>`_ or an RNN with, say, `tf.keras.layers.recurrent_v2.cudnn_lstm <https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/layers/recurrent_v2.py#L1099>`_ or the like)
* Gaussian processes (again, maybe someday, but nonparametric models also don't quite fit in to the ProbFlow framework.  Though again one could hack together a GP w/ ProbFlow, for example using `tfd.GaussianProcess <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcess>`_)
* Bayesian model comparison (also maybe someday)
* Backends other than TensorFlow/TFP and PyTorch
* Automatic variational posterior generation (`a la Pyro <http://docs.pyro.ai/en/stable/infer.autoguide.html>`_)
* Automatic reparameterizations
* Streaming inference


Issues
------

* Flipout implementation for sure has a bug - fitting a DenseRegression works w/o flipout, but not w/ it.
* Autograph fails. "Entity <blah blah> could not be transformed and will be executed as-is", where blah is "function sum" or "bound method Gamma.call of <probflow.distributions.Gamma..." etc.  Don't get the warnings when you remove the ``@tf.function`` in front of the ``train_step`` func defined in models.Model._train_step_tensorflow.  Happens both w/ CPU and GPU versions of TF 2.0.  Presumably b/c autograph doesn't handle various python features (lambda funcs, fancy list comprehensions, etc) used.  Runs about 3x slower w/o autograph optimization (just running w/ eager) on medium-sized neural net, probably will be much slower for smaller models.
* Model.metric (mae) causes too much memory usage (out of mem on colab w/ 100k sample linear regression?). Accidentally making a N^2 matrix maybe?
* Gamma distribution isn't passing the fit test (in tests/stats/test_distribution_fits)
* Add type hinting and enforcing
* Implement + test mean() for InverseGamma, Bernoulli, Categorical, and OneHotCategorical for pytorch
* Implement mixture distribution w/ pytorch backend. They're working on a MixtureSameFamily distribution for PyTorch (https://github.com/pytorch/pytorch/pull/22742) so maybe wait for that.
* Allow learning rate to be updated w/ PyTorch
* Model predictive sampling functions don't work when x is a Pandas DataFrame (because you can't expand_dims on a df)
* PyTorch pf.Distribution.mode()?
