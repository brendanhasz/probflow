# ProbFlow

[![Version Badge](https://img.shields.io/pypi/v/probflow)](https://pypi.org/project/probflow/)  [![Build Badge](https://github.com/brendanhasz/probflow/workflows/tests/badge.svg)](https://github.com/brendanhasz/probflow/actions?query=branch%3Amain)  [![Docs Badge](https://readthedocs.org/projects/probflow/badge/)](http://probflow.readthedocs.io)  [![Coverage Badge](https://codecov.io/gh/brendanhasz/probflow/branch/main/graph/badge.svg)](https://codecov.io/gh/brendanhasz/probflow)

ProbFlow is a Python package for building probabilistic Bayesian models with [TensorFlow 2.0](http://www.tensorflow.org/beta) or [PyTorch](http://pytorch.org) or [JAX](https://docs.jax.dev), performing stochastic variational inference with those models, and evaluating the models' inferences.  It provides both high-level modules for building Bayesian neural networks, as well as low-level parameters and distributions for constructing custom Bayesian models.

It's very much still a work in progress.

- **Git repository:** http://github.com/brendanhasz/probflow
- **Documentation:** http://probflow.readthedocs.io
- **Bug reports:** http://github.com/brendanhasz/probflow/issues


## Getting Started

**ProbFlow** allows you to quickly and less painfully build, fit, and evaluate custom Bayesian models (or [ready-made](http://probflow.readthedocs.io/en/latest/api/applications.html) ones!) which run on top of either [TensorFlow 2.0](http://www.tensorflow.org/beta) and [TensorFlow Probability](http://www.tensorflow.org/probability) or [PyTorch](http://pytorch.org) or or [JAX](https://docs.jax.dev).

With ProbFlow, the core building blocks of a Bayesian model are parameters and probability distributions (and, of course, the input data).  Parameters define how the independent variables (the features) predict the probability distribution of the dependent variables (the target).

For example, a simple Bayesian linear regression

$$y \sim \text{Normal}(wx + b, \sigma)$$

can be built by creating a ProbFlow Model.  This is just a class which inherits `pf.Model` (or `pf.ContinuousModel` or `pf.CategoricalModel` depending on the target type).  The `__init__` method sets up the parameters, and the `__call__` method performs a forward pass of the model, returning the predicted probability distribution of the target:

<details open>
<summary>Tensorflow</summary>

```python
import probflow as pf
import tensorflow as tf


class LinearRegression(pf.ContinuousModel):
    def __init__(self):
        self.weight = pf.Parameter(name="weight")
        self.bias = pf.Parameter(name="bias")
        self.std = pf.ScaleParameter(name="sigma")

    def __call__(self, x):
        return pf.Normal(x * self.weight() + self.bias(), self.std())


model = LinearRegression()
```

</details>

<details>
<summary>PyTorch</summary>

```python
import probflow as pf
import torch


class LinearRegression(pf.ContinuousModel):
    def __init__(self):
        self.weight = pf.Parameter(name="weight")
        self.bias = pf.Parameter(name="bias")
        self.std = pf.ScaleParameter(name="sigma")

    def __call__(self, x):
        x = torch.tensor(x)
        return pf.Normal(x * self.weight() + self.bias(), self.std())


model = LinearRegression()
```

</details>

<details>
<summary>JAX</summary>

```python
import probflow as pf


class LinearRegression(pf.ContinuousModel):
    def __init__(self):
        self.weight = pf.Parameter(name="weight")
        self.bias = pf.Parameter(name="bias")
        self.std = pf.ScaleParameter(name="sigma")

    def __call__(self, x):
        return pf.Normal(x * self.weight() + self.bias(), self.std())


model = LinearRegression()
```

</details>

Then, the model can be fit using stochastic variational inference, in *one line*:

```python
# x and y are Numpy arrays or pandas DataFrame/Series
model.fit(x, y)
```

You can generate predictions for new data:

```python
# x_test is a Numpy array or pandas DataFrame
>>> model.predict(x_test)
[0.983]
```

Compute *probabilistic* predictions for new data, with 95% confidence intervals:

```python
model.pred_dist_plot(x_test, ci=0.95)
```

![pred_dist_light](https://raw.githubusercontent.com/brendanhasz/probflow/main/docs/img/pred_dist_light.svg?sanitize=true)

Evaluate your model's performance using metrics:

```python
>>> model.metric('mse', x_test, y_test)
0.217
```

Inspect the posterior distributions of your fit model's parameters, with 95% confidence intervals:

```python
model.posterior_plot(ci=0.95)
```

![posteriors_light](https://raw.githubusercontent.com/brendanhasz/probflow/main/docs/img/posteriors_light.svg?sanitize=true)

Investigate how well your model is capturing uncertainty by examining how accurate its predictive intervals are:

```python
>>> model.pred_dist_coverage(ci=0.95)
0.903
```

and diagnose *where* your model is having problems capturing uncertainty:

```python
model.coverage_by(ci=0.95)
```

![coverage_light](https://raw.githubusercontent.com/brendanhasz/probflow/main/docs/img/coverage_light.svg?sanitize=true)

ProbFlow also provides more complex modules, such as those required for building Bayesian neural networks.  Also, you can mix ProbFlow with TensorFlow (or PyTorch!) code.  For example, even a somewhat complex multi-layer Bayesian neural network like this:

![dual_headed_net_light](https://raw.githubusercontent.com/brendanhasz/probflow/main/docs/img/dual_headed_net_light.svg?sanitize=true)

Can be built and fit with ProbFlow in only a few lines:


<details open>
<summary>Tensorflow</summary>

```python
import probflow as pf
import tensorflow as tf


class DensityNetwork(pf.ContinuousModel):
    def __init__(self, units, head_units):
        self.core = pf.DenseNetwork(units)
        self.mean = pf.DenseNetwork(head_units)
        self.std = pf.DenseNetwork(head_units)

    def __call__(self, x):
        z = tf.nn.relu(self.core(x))
        return pf.Normal(self.mean(z), tf.exp(self.std(z)))


# Create the model
model = DensityNetwork([x.shape[1], 256, 128], [128, 64, 32, 1])

# Fit it!
model.fit(x, y)
```

</details>

<details>
<summary>PyTorch</summary>

```python
import probflow as pf
import torch


class DensityNetwork(pf.ContinuousModel):
    def __init__(self, units, head_units):
        self.core = pf.DenseNetwork(units)
        self.mean = pf.DenseNetwork(head_units)
        self.std = pf.DenseNetwork(head_units)

    def __call__(self, x):
        x = torch.tensor(x)
        z = torch.nn.ReLU()(self.core(x))
        return pf.Normal(self.mean(z), torch.exp(self.std(z)))


# Create the model
model = DensityNetwork([x.shape[1], 256, 128], [128, 64, 32, 1])

# Fit it!
model.fit(x, y)
```

</details>

<details>
<summary>JAX</summary>

```python
import jax.nn
import jax.numpy as jnp
import probflow as pf


class DensityNetwork(pf.ContinuousModel):
    def __init__(self, units, head_units):
        self.core = pf.DenseNetwork(units)
        self.mean = pf.DenseNetwork(head_units)
        self.std = pf.DenseNetwork(head_units)

    def __call__(self, x):
        z = jax.nn.relu(self.core(x))
        return pf.Normal(self.mean(z), jnp.exp(self.std(z)))


# Create the model
model = DensityNetwork([x.shape[1], 256, 128], [128, 64, 32, 1])

# Fit it!
model.fit(x, y)
```

</details>

For convenience, ProbFlow also includes several [pre-built models](http://probflow.readthedocs.io/en/latest/api/applications.html) for standard tasks (such as linear regressions, logistic regressions, and multi-layer dense neural networks).  For example, the above linear regression example could have been done with much less work by using ProbFlow's ready-made LinearRegression model:

```python
model = pf.LinearRegression(x.shape[1])
model.fit(x, y)
```

And a multi-layer Bayesian neural net can be made easily using ProbFlow's ready-made DenseRegression model:

```python
model = pf.DenseRegression([x.shape[1], 128, 64, 1])
model.fit(x, y)
```

Using parameters and distributions as simple building blocks, ProbFlow allows for the painless creation of more complicated Bayesian models like [generalized linear models](http://probflow.readthedocs.io/en/latest/examples/glm.html), [deep time-to-event models](http://probflow.readthedocs.io/en/latest/examples/time_to_event.html), [neural matrix factorization](http://probflow.readthedocs.io/en/latest/examples/nmf.html) models, and [Gaussian mixture models](http://probflow.readthedocs.io/en/latest/examples/gmm.html).  You can even mix [probabilistic and non-probabilistic models](http://probflow.readthedocs.io/en/latest/examples/neural_linear.html)!  Take a look at the [examples](http://probflow.readthedocs.io/en/latest/examples/examples.html) and the [user guide](http://probflow.readthedocs.io/en/latest/user_guide/user_guide.html) for more!


## Installation

If you have an existing project, just add `probflow` to your pyproject.toml
file's dependencies section.

Or, install with pip.  if you already have your desired backend installed
(i.e. Tensorflow/TFP or PyTorch or JAX), then you can just do:

```bash
pip install probflow
```

Or, to install both ProbFlow and your desired backend,

<details open>
<summary>Tensorflow</summary>

```bash
pip install probflow[tensorflow]
```

</details>

<details>
<summary>PyTorch</summary>

```bash
pip install probflow[pytorch]
```

</details>

<details>
<summary>JAX</summary>

Unlike TensorFlow and PyTorch, JAX requires a GPU-specific installation for GPU support.
If you are only planning on running ProbFlow on the CPU, you can just use the standard JAX installation:

```bash
pip install probflow[jax]
```

But if you want to run on your GPU, also install the appropriate JAX CUDA package:

```bash
pip install probflow[jax] jax[cuda13]
```

</details>


## Jupyter Notebook

To run ProbFlow code in a Jupyter Notebook, you can either install ProbFlow into your
existing kernel environment (see above), or clone the repo and run the notebook server
from there.  There is a convenient Makefile command for this. You can start a notebook
server with ProbFlow and a given backend pre-installed by running:

<details>
<summary>Tensorflow</summary>

```bash
git clone git@github.com:brendanhasz/probflow.git
cd probflow
make notebook-server BACKEND=tensorflow
```

</details>

<details>
<summary>PyTorch</summary>

```bash
git clone git@github.com:brendanhasz/probflow.git
cd probflow
make notebook-server BACKEND=pytorch
```

</details>

<details>
<summary>JAX</summary>

```bash
git clone git@github.com:brendanhasz/probflow.git
cd probflow
make notebook-server BACKEND=jax
```

</details>

Then open your web browser and navigate to [http://localhost:8888](http://localhost:8888) to access the Jupyter Notebook server.


## Support

Post bug reports, feature requests, and tutorial requests in [GitHub issues](http://github.com/brendanhasz/probflow/issues).


## Contributing

[Pull requests](http://github.com/brendanhasz/probflow/pulls) are totally welcome!  Any contribution would be appreciated, from things as minor as pointing out typos to things as major as writing new applications and distributions.


## Why the name, ProbFlow?

Because it's a package for probabilistic modeling, and it was built on TensorFlow.  ¯\\\_(ツ)_/¯
