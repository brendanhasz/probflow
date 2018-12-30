Would be ideal to have, basically, Keras for Bayesian models, built on tfp
With added support for viewing/drawing from posteriors, computing predictive distributions, viewing uncertainty calibration metrics, etc

Let's call it bk for now ("bayesian keras"... that'll have to change, obvi)

TODO short term:
- docs for BaseLayer
- Variable (don't worry about lb and ub for now)
- BaseModel (except for the stuff_by funcs and the cdf funcs)
- BaseModel docs
- Input layer
- Test and debug a simple 1d linear model
- More test which cover distributions, layers, and core elements you've written (ensure right shapes etc)
- Tests that ensure layer or model can take any (valid) combo of types as args
- Write/docs for rest of core
- Dense layer!
- Test and compare to stan or edward
- Tests for dense
- Sequential layer
- Tests
- Models which only use Dense
- Tests

TODO: mid term:
- Bernoulli and Poisson dists
- Models which use them (Classifiers)
- Tests

TODO long term:
- Conv layers
- Pooling layer
- Embedding layer
- Nueral Matrix Factorization
- Tensorflow graph / dashboard
- Model comparison
- Mixture distribution

## bk.variables
Just a way go get raw variational variables
- Variable (returns a tfp.normal, shape can be >1)
- PositiveVariable
- BoundedVariable
alternatively, just do Variable(..., lb=0) or Variable(..., lb=0, ub=1)
and should be able to set the estimator: Variable(..., estimator='flipout') #or 'random'?

## bk.layers
Layers output a tf.Tensor, whereas variables, distributions, and models output a tfd.distribution
(Built with tfp's flipout estimators)
- BaseLayer (from which all other layers inherit)
- Sequential
- Dense (default activation is relu)
- Embedding (default is to not use prob dist for embeddings)
- Conv1D
- Conv2D
- LSTM? Maybe put that on the Todo list...
All should have an in=... arg which specifies their input(s)

## bk.distributions
(Basically aliases tfp distributions, also add LogNormal and LogitNormal?)
-  Normal (defaults are loc=0, scale=1)
-  Bernoulli

## bk.models
(Pre built models)
- BaseModel (has code for fit() methods, _init_(), etc and everything that's the same which variables, distributions, and models inherit)
- LinearRegression   #Regressors infer dim of outputs from dim of y
- LogisticRegression #Classifiers infer dim of outputs from nunique of y
- DenseRegression (just specify list of num units in each layer)
- DenseClassifier
- Conv1dRegression (2 args: 1st list is num conv layers, 2nd list is num dense layers)
- Conv1dClassifier (ditto)
- Conv2dRegression (ditto)
- Conv2dClassifier (ditto)
- DenseAutoencoder (last num is num latent units)
- Conv1dAutoencoder 
- Conv2dAutoencoder
- Recommendation one?
- Bayesian Correlation? ... nah that doesn't really fit into the framework does it?  well yeah I guess it does, pass it x and y...

## Fitting
Use elbo loss by default.

Should be able to do:

predictions = Sequential ([
    Dense (32),
    Dense (32),
    Dense(1)
])
noise_std = Variable(lb=0)
model = Normal(predictions, noise_std)
model.fit(x, y, 
          batch_size=128, epochs=10,
          optimiser='adam', #the default
          learning_rate=0.05, #should set a reasonable default
          metrics='mae') #should set a reasonable default based off what y data looks like

Fit func should also have a validation_split arg (which defines what percentage of the data to use as validation data, along w/ shuffle arg, etc)
Fit func should automatically normalize the input data and try to transform it (but can disable that w/ auto_normalize=False, auto_transform=False)
And convert to float32s (but can change w/ dtype=tf.float64 or something)
Also should have callbacks like let's to do early stopping etc
Also have a monitor param which somehow lets you monitor the value of parameters over training?
And make sure to make it sklearn-compatible (ie you can put it in a pipeline and everything)
should have a show_tensorboard option (default is False) https://www.tensorflow.org/guide/summaries_and_tensorboard


## Priors
Should have some way to set priors?
And the layers and models and everything should have sensible default priors (like normal dist w/ mu=0 and sig=1)


## Keeping track of the loss
in build() will have to compute the log_loss and set self.log_loss to that
and when computing that, make sure to add the losses of all children too
Still have to figure out whether the -elbo loss is just the log posterior prob, or do we need to add the kl term?


## Evaluation of the posterior
Then should also have ways to draw from posterior dist,eg

- model.predict(x) #take avg/median/mode? of draws from posterior
- model.residuals(x, y) #compute residuals on validation data (returns data, plots by default but can control w/ plot=False)
- model.plot_residuals(x, y) #plot the residual distribution
- model.metrics(x, y, metric_list) #compute metrics on validation data
- model.posterior() #somehow specify which params you want to sample posterior for?
        Should return a dict w/ {"param_name": samples, ...}
        And params could be named w/ name=... arg (to variable, distributions, layers, etc funcs)
        Pass a list of strs to posterior() to limit what it returns to only those
-  model.predictive_distribution(x) #return the predictive distribution (samples) for features x
-  model.predictive_prc(x, y) #return the percentile at which each y falls on it's predictive distribution given x
-  model.pred_dist_covered(x, y, prc) #returns vector of length y, 1 if y_i is in middle prc of predictive dist, else 0
-  model.pred_dist_coverage(x, y, prc) #returns single value, percentage of coverage, ideally is = prc sick was passed. Prc can be a vector to make a calibration curve
-  model.calibration_curve(x, y)? #returns calibration curve against ideal line (plots by default but can turn off w/ plot=False arg)
-  model.coverage_by(x, y, x_by, prc) #plots the prc coverage across an x_by variable(s). x can be a matrix/vector or a list of them.  If a list it will make a plot by each. If x_by is 1D, plots a line, if 2D an image plot w RGB color map (r when coverage is too low, g when coverage is just right, and blue when coverage is too high, but fade to Black w less data, may want to use vaex)

(funcs which draw from posterior can take a num_samples arg for # of samples)

But, most of these will only work with y-variables which are continuous?


## Implementation notes
Each of the above models, layers, variables, and distributions should be a class
All (except layers) inherit from BaseModel which has fit() and posterior-related methods (and other attribs)
So they'll all have a .build() method (even the layers - BaseL
    build() method recursively calls build() methods of objs this one depends on
    build() returns a tfd.distribution (except for layers??? those return tensors, will have to check if it's a layer obj or not)
    and you'll know from the tfd.distribution obj what shape it is
    will have to pass the x data placeholder to each call to build? so when we arrive @ the lowest level can have access to it
    so will have to have .built and .built_model attribs which are None until build() is called
Each class will also have to have a .fit() method too (maybe can be implemented in BaseModel class which all inherit from)
    
Behind the scenes bk turns stuff like
    Normal(Dense(...), Normal(...))
into
    tfd.Normal(tfp.DenseFlipout(...)(x_in), tfd.Normal(...).sample())
or something
So for each of a things parameters, it detects if that param is a tfd.distribution object, 
    if so use .sample(sample_size=batch_size)



# TODOs: 

## Tensorflow graph
Make a model.tensorboard_graph(...same args as fit?...) method or something
https://www.tensorflow.org/guide/graph_viz

## Handling data inputs
Don't worry about this till you get the main stuff working...
To manually feed in data, use Input layer?
which takes list of ints (cols of x for which this Input obj should represent)
  or list of strings (if x is a pandas df)
    i.e., use tf.placeholders where you want them in 
    can pass dict between placeholders and their data/dataset_iterators 
Layers and models have an in=... arg which specifies the input to their first layer?
If it's empty (None) then assume the x values are the input
For example, for a linear regression you can do  
  predictions = Dense(1)
  noise_std = Variable(lb=0)
  model = Normal(predictions, noise_std)
  model.fit(x, y, ...)
or, to explicitly state where the x values should be used,
  x_in = tf.placeholder(tf.float32, shape=(4,))
  predictions = Dense(1, input=x_in)
  noise_std = Variable(lb=0)
  model = Normal(predictions, noise_std)
  model.fit({x_in: data, 
Or, use dataset iterators (and pass the training and validation handles)

## Random effects and multilevel models
Also, should allow for random effects, mixed effects (just the results of a fixed effects net plus the results of a random effects net) and also hierarchical/multilevel models (where random effect variables are nested).
Ie for random effects there's an over all dist of weights, but each subject/group has their own weight distributions which are drawn from pop dist
Use the reparam trick?
And should be able to make multilevel model with that: eg individuals drawn from schools (in fact comparing to the 8 schools example in r would be good way to test that it works)
Perhaps make a RandomVariable() which takes a slice of the x_values placeholder? (as individual/group id or whatever)

## Handling Bijectors
???

## Mixture distribution
A continuous distribution which takes a list of other distrbutions.

## Model comparison
somehow.  AIC/BIC/DIC?  Could you do LOO?
I mean.  Or just use held-out log posterior prob...



# EXAMPLES

## Linear Regression
```python
predictions = Dense(1)
noise_std = Variable(lb=0)
model = Normal(predictions, noise_std)
model.fit(x, y)
```
or, more simply:
```python
model = LinearRegression()
model.fit(x,y)
```

## Logistic Regression
```python
logits = Dense(1)
model = Bernoulli(logits)
model.fit(x,y)
```
or, using the bk.model,
```python
model = LogisticRegression()
model.fit(x,y)
```

## Dense Neural Network Regression
```python
predictions = Sequential([
    Dense(128),
    Dense(64),
    Dense(32),
    Dense(1)
])
noise_std = Variable(lb=0)
model = Normal(predictions, noise_std)
model.fit(x,y)
```
or, with the bk.model,
```python
model = DenseRegression([128, 64, 32])
model.fit(x,y)
```

## Convolutional Neural Network 
MNIST example
TODO

## Dual-module deep neural net which estimates predictions and aleatoric uncertainty separately
```python
predictions = DenseRegression([128, 64, 32])
noise_std = DenseRegression([128, 64, 32])
model = Normal(predictions, noise_std)
model.fit(x,y)
```

