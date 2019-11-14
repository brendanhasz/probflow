Backlog
=======

This page has a list of planned improvements, in order of when I plan to get
to them (the backlog).  If you're interested in tackling one of them, I'd be 
thrilled!  `Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!  There's also a list of things which I think are out of
scope for the package, and for actual bugs/issues see the `issues page <https://github.com/brendanhasz/probflow/issues>`_


Backlog
-------

* User guide
* Examples
* Docs for everything implemented so far
* Fix some issues (below)
* Release 2.0.0
* Get tf autograph/tf.function and pytorch torchscript/trace working!!!
* Add ability to use >1 MC samples from variational distribution per batch
* Add HiddenMarkovModel and GaussianProcess distributions + examples
* Bayesian decision making example
* Finish pred_dist_plot for Discrete and Categorical Model, as well as calibration_curve for categorical.
* Make Parameters have an option for how to plot their posteriors/priors (ie use plot_dist, plot_discrete_dist, or plot_categorical_dist), and set defaults that make sense for parameters (e.g. default for Categorical param should be plot_categorical_dist)
* Add model saving/loading (Model.save_model and load_model)
* Summary method for Modules + Models which show hierarchy of modules/parameters
* Make Module.trainable_variables return tf.Variables which are properties of module+sub-modules as well (and not neccesarily in parameters, also allow embedding of tf.Modules?)
* Add kwarg to Parameter (mc_kl_estimate=False or somesuch) which if true will use MC samples to estimate KL divergence between parameter's prior and variational posterior (to allow for arbitrary posterior/prior choices)
* Convolutional modules
* Add more consistent type hinting and enforcing (perhaps via `enforce.runtime_validation <https://github.com/RussBaz/enforce>`_ or `pytypes.typechecked <https://github.com/Stewori/pytypes>`_ or `typeguard.typechecked <https://github.com/agronholm/typeguard>`_ ).  Or maybe just remove all type hinting and just have it in the docstrings...


Out of scope
------------

Things which would conceivably be cool, but I think are out of scope for the
package:

* Forms of inference aside from stochastic variational inference (e.g. MCMC, Stein methods, Langevin dynamics, Laplace approximation, importance sampling, expectation maximization, particle filters, finding purely the MAP/ML estimate, Bayesian estimation via dropout, etc...)
* Support for sequence models like RNNs and HMMs (though maybe someday will support that, but ProbFlow is aimed more towards supporting tabular/matrix data as opposed to sequence data.  Note that you could probably hack together a HMM using `tfd.HiddenMarkovModel <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel>`_ or an RNN with, say, `tf.keras.layers.recurrent_v2.cudnn_lstm <https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/layers/recurrent_v2.py#L1099>`_ or the like)
* Gaussian processes (again, maybe someday, but nonparametric models also don't quite fit in to the ProbFlow framework.  Though again one could hack together a GP w/ ProbFlow, for example using `tfd.GaussianProcess <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcess>`_).  See `this issue <https://github.com/brendanhasz/probflow/issues/7>`_ for more info.
* Bayesian model comparison (also maybe someday)
* Backends other than TensorFlow/TFP and PyTorch
* Automatic variational posterior generation (`a la Pyro <http://docs.pyro.ai/en/stable/infer.autoguide.html>`_).
* Automatic reparameterizations
* Streaming inference
