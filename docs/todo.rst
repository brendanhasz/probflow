Backlog
=======

This page has a list of planned improvements, in order of when I plan to get
to them (the backlog).  If you're interested in tackling one of them, I'd be 
thrilled!  `Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!  There's also a list of things which I think are out of
scope for the package, and for actual bugs/issues see the `issues page <https://github.com/brendanhasz/probflow/issues>`_


Backlog
-------

* Get tf autograph/tf.function and pytorch torchscript/trace working!!!
    - Replace lambda functions in codeset: In `pf.core.ops`, define an `identity` function which literally just returns its one arg. Like `lambda x: x`, but a defined function so tensorflow can see it + compile w/ it.  Then in `Parameter.__init__`, make `self.transform = O.identity if transform is None else transform`.  That way you can specify `transform=None` for parameters (in their constructors and when calling) and tensorflow will be able to compile (because it doesn't like lambda functions, and all your transforms are defined as lambda functions currently).  While you're at it define actual functions for the other transforms used:  `Parameter`'s transform can be None, `ScaleParameter`'s transform can just be `O.sqrt` instead of `lambda x: O.sqrt(x)`, `Categorical`'s transform can be None, `DirichletParameter`'s transform can be None, `BoundedParameter should define a closure function in `__init__` instead of using a lambda function.  Also use `O.identity` to replace all the `lambda x: x`'s in the var_transforms.  Oof and will have to update all of that in the tests...  And `pf.applications.DenseNetwork` uses `lambda x: x` for the activation of the last layer, replace w/ `O.identity`.
    - Fix all the list/dict comprehensions tensorflow can't handle/compile
* For `pf.applications.LinearRegression` with `heteroskedastic=True` make bias of shape `[1, 2]` and use the 2nd for the std dev bias (currently the std dev is missing a bias term!)
* User guide
* Examples
* Docs for everything implemented so far
* Release 2.0.0
* Add a `probabilistic` kwarg (True or False) to Dense, DenseNetwork, Embedding, and BatchNormalization modules. That way you can pretty easily do, say, a non-probabilistic net with a probabilistic linear layer on top (cite that google paper), and set whether you want your embeddings + normalizations probabilistic or not.  Allow the user to specify both the probabilistic kwarg and the posterior+initializer kwargs (which you can already do in BatchNormalization and Embedding, but you'll have to add it to Dense), and any non-default values of the posteriors and initializers passed to constructor, override what the probabilistic kwarg has set.
* Then add a "Mixing Probabilistic and Deterministic Models" section to examples
* Add a `CenteredParameter`, which should use QR reparameterization for vector of parameters centered at zero. Have a `center_by` kwarg (one of 'all', 'column', 'row') which determines how they're centered.  'all' means the sum of all elements, regardless of shape, is 0.  'column' means the sum of each column is 0, and 'row' means the sum of each row is 0.  For 'all', get a prod(shape)-length vector via the QR reparameterization, then reshape into the correct shape.  For 'column' and 'row', only allow 2d shape, and can matrix multiply the A from the QR reparameterization by a matrix, then transpose for row. Can do that in the transform function, and have appropriate priors such that resulting parameters have prior ~ normal(0, 1).  And make sure to mention that the prior is fixed in the docs.
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
* Add a `set_priors_to_posteriors` (or perhaps something more elegant... `bayesian_update`? Just `update`?) function to `pf.models.Model` which sets the prior distributions to the current value of the posterior distributions, to allow Bayesian updating / streaming inference / incremental updates.  Should be relatively straightforward I think :thinking: Like, literally just `for p in self.parameters: p.prior = p.posterior`

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
