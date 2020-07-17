Backlog
=======

See the
`github projects page <https://github.com/brendanhasz/probflow/projects/1>`_
for a list of planned improvements (in the "To do" column, in order of priority
from highest at the top to lowest at the bottom), as well as what's being
worked on currently in the "In progress tab".

If you're interested in tackling one of them, I'd be thrilled!
`Pull requests <https://github.com/brendanhasz/probflow/pulls>`_
are totally welcome!  Also take a look at the :doc:`dev_guide/dev_guide`.


Out of scope
------------

Things which would conceivably be cool, but I think are out of scope for the
package:

* Forms of inference aside from stochastic variational inference (e.g. MCMC, Stein methods, Langevin dynamics, Laplace approximation, importance sampling, expectation maximization, particle filters, finding purely the MAP/ML estimate, Bayesian estimation via dropout, etc...)
* Support for sequence models like RNNs and HMMs (though maybe someday will support that, but ProbFlow is aimed more towards supporting tabular/matrix data as opposed to sequence data.  Note that you could probably hack together a HMM using `tfd.HiddenMarkovModel <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HiddenMarkovModel>`_ or an RNN with, say, `tf.keras.layers.recurrent_v2.cudnn_lstm <https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/layers/recurrent_v2.py#L1099>`_ or the like)
* Gaussian processes (again, maybe someday, but nonparametric models also don't quite fit in to the ProbFlow framework.  Though again one could hack together a GP w/ ProbFlow, for example using `tfd.GaussianProcess <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/GaussianProcess>`_).  See `this issue <https://github.com/brendanhasz/probflow/issues/7>`_ for more info.  If you want to use Gaussian processes with TensorFlow, I'd suggest `GPflow <https://github.com/GPflow/GPflow>`_, or `GPyTorch <https://gpytorch.ai>`_ for PyTorch.
* Bayesian networks (though again, could probably manually hack one together in ProbFlow, at least one with a fixed DAG structure where you just want to infer the weights)
* Bayesian model comparison (also maybe someday)
* Backends other than TensorFlow/TFP and PyTorch
* Automatic variational posterior generation (`a la Pyro <http://docs.pyro.ai/en/stable/infer.autoguide.html>`_).
* Automatic reparameterizations
