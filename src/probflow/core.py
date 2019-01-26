"""Abstract classes.

TODO: more info...

----------

"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

__all__ = [
    'REQUIRED',
    'BaseParameter',
    'BaseLayer',
    'BaseDistribution',
    'ContinuousDistribution',
    'DiscreteDistribution',
]


# Sentinel object for required arguments
REQUIRED = object()



class BaseObject(ABC):
    """Abstract probflow object class (used as an implementation base)"""

    def __add__(self, other):
        """Add this layer to another layer, parameter, or value."""
        from .layers import Add
        return Add(self, other)


    def __sub__(self, other):
        """Subtract from this layer another layer, parameter, or value."""
        from .layers import Sub
        return Sub(self, other)


    def __mul__(self, other):
        """Multiply this layer by another layer, parameter, or value."""
        from .layers import Mul
        return Mul(self, other)


    def __truediv__(self, other):
        """Divide this layer by another layer, parameter, or value."""
        from .layers import Div
        return Div(self, other)


    def __abs__(self):
        """Take the absolute value of the input to this layer."""
        from .layers import Abs
        return Abs(self)


    def __neg__(self):
        """Take the negative of the input to this layer."""
        from .layers import Neg
        return Neg(self)


    def __matmul__(self, other):
        """Matrix multiply this layer by another, per PEP 465."""
        from .layers import Matmul
        return Matmul(self, other)



class BaseParameter(BaseObject):
    """Abstract parameter class (used as an implementation base)"""
    pass



class BaseLayer(BaseObject):
    """Abstract layer class (used as an implementation base)

    This is an abstract base class for a layer.  Layers are objects which take
    other objects as input (other layers, parameters, or tensors) and output a
    tensor.

    TODO: more...


    Required attributes and methods
    -------------------------------
    An inheriting class must define the following properties and methods:
    * `_default_args` (attribute)
    * `_build` (method)
    * `_log_loss` (method)

    The `_default_args` attribute should contain a dict whose keys are the names
    of the layer's arguments, and whose values are the default value of each
    argument.  Setting an argument's value in `_default_args` to `None` causes
    that argument to be mandatory (TypeError if that argument's value is not
    specified when instantiating the class).

    The `_build` method should return a `Tensor` which was built from this
    layer's arguments. TODO: more details...

    The `_log_loss` method should return the log loss incurred by this layer.

    TODO: more...


    See Also
    --------
    func : why you should see it


    Notes
    -----
    TODO: Docs...


    Examples
    --------

    We can define a layer which adds two arguments like so::

        class Add(BaseLayer):

            self._default_args = {
                'a': None,
                'b': None
            }

            def _build(self, args, data):
                return args['a'] + args['b']

            def _log_loss(self, obj, vals):
                return 0

    then we can use that layer to add two other layers or tensors::

        x = Input()
        b = Parameter()
        mu = Add(x, b)

    For more examples, see :class:`.Add`, :class:`.Sub`, :class:`.Mul`,
    :class:`.Div`, :class:`.Abs`, :class:`.Exp`, and :class:`.Log`.

    """


    # Default keyword arguments for a layer
    _default_kwargs = dict()


    # Layer arguments and their default values
    _default_args = {
        'input': REQUIRED
    }


    def __init__(self, *args, **kwargs):
        """Construct layer.

        TODO: docs. Mention that actually building the tf graph is
        delayed until build() or fit() is called.
        """

        # Set layer arguments, using args, kwargs, and defaults
        self.args = dict()
        for ix, arg in enumerate(self._default_args):
            if ix < len(args):
                self.args[arg] = args[ix]
            elif arg in kwargs:
                self.args[arg] = kwargs[arg]
            else:
                self.args[arg] = self._default_args[arg]

        # Ensure all required arguments have been set
        for val in self.args.values():
            if val is REQUIRED:
                raise TypeError('required arg(s) were not set. '+
                                type(self).__name__+' requires args: '+
                                ', '.join(self._default_args.keys()))

        # Ensure all arguments are of correct type
        for arg in self.args:
            if not self._arg_is('valid', self.args[arg]):
                msg = ('Invalid type for ' + type(self).__name__ +
                       ' argument ' + arg +
                       '.  Arguments to a layer must be one of: ' +
                       'int, float, np.ndarray, ' +
                       'tf.Tensor, tf.Variable, ' +
                       'or a probflow layer or parameter.')
                raise TypeError(msg)

        # Set layer kwargs
        self.kwargs = dict()
        for ix, kwarg in enumerate(self._default_kwargs):
            if len(args) > (len(self._default_args)+ix): #leftover args!
                self.kwargs[kwarg] = args[len(args)+ix]
            elif kwarg in kwargs:
                self.kwargs[kwarg] = kwargs[kwarg]
            else:
                self.kwargs[kwarg] = self._default_kwargs[kwarg]

        # Set attribs for the built layer and fit state
        self.built_obj = None
        self.built_args = None
        self.mean_obj = None
        self.mean_args = None
        self.is_fit = False
        self.log_loss = 0


    @abstractmethod
    def _build(self, args, data, batch_shape):
        """Build layer.

        Inheriting class must define this method by building the layer for that
        class.  Should return a `Tensor` or a `tfp.distribution` using the
        layer arguments in args (a dict).

        TODO: docs...

        """
        pass


    def _build_mean(self, args, data, batch_shape):
        """Build the layer with mean parameters.

        TODO: docs. default is to just do the same thing as _build

        """
        return self._build(args, data, batch_shape)


    def _log_loss(self, vals):
        """Compute the log loss incurred by this layer.

        TODO: docs... default is no loss but can override when there is

        """
        return 0


    def _mean_log_loss(self, vals):
        """Compute the log loss incurred by this layer with mean parameters.

        TODO: docs... default is no loss but can override when there is

        """
        return 0


    def _kl_loss(self):
        """Compute the loss due to posterior divergence from priors.

        TODO: docs... default is no loss but can override when there is

        """
        return 0


    def _arg_is(self, type_str, arg):
        """Return true if arg is of type type_str."""
        if type_str == 'tensor_like':
            return isinstance(arg, (int, float, np.ndarray,
                                    tf.Tensor, tf.Variable))
        elif type_str == 'distribution':
            return isinstance(arg, BaseDistribution)
        elif type_str == 'layer':
            return isinstance(arg, BaseLayer)
        elif type_str == 'parameter':
            return isinstance(arg, BaseParameter)
        elif type_str == 'valid': #valid input to a layer
            return (not isinstance(arg, BaseDistribution) and
                    isinstance(arg, (int, float, np.ndarray,
                                     tf.Tensor, tf.Variable,
                                     BaseLayer, BaseParameter)))
        else:
            raise TypeError('type_str must a string, one of: number, tensor,' +
                            ' tensor_like, model, layer, or valid')


    def build(self, data, batch_shape):
        """Build this layer's arguments and loss, and then build the layer.

        TODO: actually do docs for this one...

        """

        # Store a list of all parameters in the model
        self._parameters = self._parameter_list()

        # Build each of this layer's arguments.
        self.built_args = dict()
        self.mean_args = dict()
        for arg_name, arg in self.args.items():
            if isinstance(arg, int):
                self.built_args[arg_name] = float(arg)
                self.mean_args[arg_name] = float(arg)
            elif self._arg_is('tensor_like', arg):
                self.built_args[arg_name] = arg
                self.mean_args[arg_name] = arg
            elif self._arg_is('layer', arg) or self._arg_is('parameter', arg):
                arg.build(data, batch_shape)
                self.built_args[arg_name] = arg.built_obj
                self.mean_args[arg_name] = arg.mean_obj

        # Sum the losses of this layer's arguments
        self.samp_loss_sum = 0 #log posterior probability of sample model
        self.mean_loss_sum = 0 #log posterior probability of mean model
        self.kl_loss_sum = 0 #sum of KL div between variational post and priors
        for arg_name, arg in self.args.items():
            if self._arg_is('tensor_like', arg):
                pass #no loss incurred by data
            elif self._arg_is('layer', arg):
                self.samp_loss_sum += arg.samp_loss_sum
                self.samp_loss_sum += arg._log_loss(arg.built_obj)
                self.mean_loss_sum += arg.mean_loss_sum
                self.mean_loss_sum += arg._mean_log_loss(arg.mean_obj)
                self.kl_loss_sum += arg.kl_loss_sum + arg._kl_loss()
            elif self._arg_is('parameter', arg):
                self.samp_loss_sum += arg._log_loss
                self.mean_loss_sum += arg._mean_log_loss
                self.kl_loss_sum += arg._kl_loss

        # Build this layer's sample model and mean model
        self.built_obj = self._build(self.built_args, data, batch_shape)
        self.mean_obj = self._build_mean(self.mean_args, data, batch_shape)


    def _parameter_list(self):
        """Get a list of parameters in this layer or its arguments."""
        params = []
        for arg in self.args:
            if isinstance(self.args[arg], BaseLayer):
                params += self.args[arg]._parameter_list()
            elif isinstance(self.args[arg], BaseParameter):
                params += [self.args[arg]]
        return params


    def __str__(self, prepend=''):
        """String representation of a layer (and all its args)."""

        # Settings
        max_short = 40 #max length of a short representation
        ind = '  ' #indentation to use

        # Make string representation of this layer and its args
        self_str = self.__class__.__name__
        arg_strs = dict()
        for arg in self._default_args:
            if isinstance(self.args[arg], (int, float)):
                arg_strs[arg] = str(self.args[arg])
            elif isinstance(self.args[arg], (np.ndarray)):
                arg_strs[arg] = 'np.ndarray shape='+str(self.args[arg].shape)
            elif isinstance(self.args[arg], (tf.Tensor, tf.Variable)):
                arg_strs[arg] = self.args[arg].__str__()
            elif isinstance(self.args[arg], (BaseParameter, BaseLayer)):
                tstr = self.args[arg].__str__(prepend=prepend+2*ind)
                if len(tstr) < max_short:
                    arg_strs[arg] = tstr
                else:
                    arg_strs[arg] = '\n'+tstr
            else:
                arg_strs[arg] = '???'

        # Try a short representation
        short_args = [a+' = '+arg_strs[a] for a in self._default_args]
        short_str = self_str+'('+', '.join(short_args)+')'
        if len(short_str) < max_short:
            return short_str

        # Use a longer representation if the shorter one failed
        return '\n'.join([prepend+self_str] +
                         [prepend+ind+a+' = '+arg_strs[a] 
                          for a in self._default_args])



class BaseDistribution(BaseLayer):
    """Abstract distribution class (used as an implementation base)

    TODO: More info...
    talk about how a model defines a parameterized probability distribution
    which you can call fit on

    """


    # Posterior distribution parameter bounds (lower, upper)
    _post_param_bounds = {
        'input': (None, None)
    }


    def _log_loss(self, vals):
        """Compute the log loss ."""
        return self.built_obj.log_prob(vals)


    def _mean_log_loss(self, vals):
        """Compute the log loss ."""
        return self.mean_obj.log_prob(vals)


    def fit(self, x, y,
            dtype=tf.float32,
            batch_size=128,
            epochs=100,
            optimizer='adam',
            learning_rate=0.01,
            metrics=[],
            verbose=True,
            validation_split=0,
            validation_shuffle=True,
            shuffle=True):
        """Fit model.

        TODO: Docs...

        Parameters
        ----------
        x : |ndarray|
            Independent variable values of the dataset to fit (aka the 
            "features").
        y : |ndarray|
            Dependent variable values of the dataset to fit.
        dtype : TODO
            TODO
        batch_size : int
            TODO
        epochs : int
            TODO
        optimizer : TODO
            TODO
        learning_rate : float
            TODO
        metrics : str or list of str
            Metrics to evaluate each epoch.  To evaluate multiple metrics, 
            pass a list of strings, where each string is a different metric.
            Available metrics:

            * 'acc': accuracy
            * 'accuracy': accuracy
            * 'mse': mean squared error
            * 'sse': sum of squared errors
            * 'mae': mean absolute error

            Default = empty list
        verbose : bool
            Whether to print progress and metrics throughout training.
            Default = True
        validation_split : float between 0 and 1
            Proportion of the data to use as validation data.
            Default = 0
        validation_shuffle : bool
            Whether to shuffle which data is used for validation.
            Default = True
        shuffle : bool
            Whether to shuffle the training data before each trainin epoch.
            Default = True


        TODO: brief math about variational inference :ref:`[1] <ref_bbb>`
        (just to say that the loss used is -ELBO = KL - log_likelihood)


        References
        ----------
        .. _ref_bbb:
        .. [1] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and 
            Daan Wierstra. Weight uncertainty in neural networks. 
            *arXiv preprint*, 2015. http://arxiv.org/abs/1505.05424


        """

        # Check input data size matches
        if x.shape[0] != y.shape[0]:
            raise ValueError('x and y do not have same number of samples')

        # TODO: add support for pandas arrays

        # TODO: how to support integer columns?

        # Split data into training and validation data
        if validation_split > 0:
            if validation_shuffle:
                train_ix = np.random.rand(x.shape[0]) > validation_split
            else:
                train_ix = np.arange(x.shape[0]) > (validation_split*x.shape[0])
            val_ix = ~train_ix
            x_train = x[train_ix, ...]
            y_train = y[train_ix, ...]
            x_val = x[val_ix, ...]
            y_val = y[val_ix, ...]
        else:
            x_train = x
            y_train = y
            x_val = x
            y_val = y

        # Store pointer to training data
        self._x_train = x_train
        self._y_train = y_train

        # Data placeholders
        N = x_train.shape[0]
        x_shape = list(x_train.shape)
        y_shape = list(y_train.shape)
        x_shape[0] = None
        y_shape[0] = None
        x_data = tf.placeholder(dtype, x_shape)
        y_data = tf.placeholder(dtype, y_shape)
        self._initialize_shuffles(N, epochs, shuffle)

        # Batch shape
        batch_size_ph = tf.placeholder(tf.int32, [1])

        # Store placeholders
        self._batch_size_ph = batch_size_ph
        self._x_ph = x_data
        self._y_ph = y_data

        # Recursively build this model and its args
        self.build(x_data, batch_size_ph)

        # Set up TensorFlow graph for per-sample losses
        self.log_loss = (self.samp_loss_sum +  #size (batch_size,)
                         self._log_loss(y_data))
        self.mean_log_loss = (self.mean_loss_sum + #size (batch_size,)
                              self._mean_log_loss(y_data))
        self.kl_loss = tf.cast(self.kl_loss_sum + self._kl_loss(), dtype)

        # TODO: uuhh do you ever actually need log_loss?
        # that's just the log posterior prob of *samples* from the model?
        # the only place you would need that is if you called
        # log_prob (below) with distribution=True and individually=True...

        # ELBO loss function
        log_likelihood = tf.reduce_mean(self.built_obj.log_prob(y_data))
        kl_loss = tf.reduce_sum(self.kl_loss) / N
        elbo_loss = kl_loss - log_likelihood

        # TODO: determine a good default learning rate?

        # TODO: set optimizer based on optimizer arg

        # Optimizer
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(elbo_loss)

        # Create the TensorFlow session and assign it to each parameter
        self._session = tf.Session()
        for param in self._parameters:
            param._session = self._session

        # Initializers
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self._session.run(init_op)

        # Fit the model
        self.is_fit = True
        n_batch = int(np.ceil(N/batch_size)) #number of batches per epoch
        print_batches = int(np.ceil(n_batch/10)) #info each print_batches batch
        for epoch in range(epochs):

            # Print progress
            if verbose:
                print('Epoch %d / %d' % (epoch, epochs))

            # Train on each batch in this epoch
            for batch in range(n_batch):
                b_x, b_y = self._generate_batch(x_train, y_train, 
                                                epoch, batch, batch_size)
                self._session.run(train_op,
                                  feed_dict={x_data: b_x,
                                             y_data: b_y,
                                             batch_size_ph: [b_x.shape[0]]})

                # Print progress
                if verbose and batch % print_batches == 0:
                    print("  Batch %d / %d (%0.1f)\r" %
                          (batch+1, n_batch, 100.0*batch/n_batch), end='')

            # Evaluate metrics
            print(60*' '+"\r", end='')
            if metrics:
                md = self.metrics(x_val, y_val, metrics)
                print('  '+(4*' ').join([m+': '+str(md[m]) for m in md]))

        # Finished!
        print('Done!')


    def _initialize_shuffles(self, N, epochs, shuffle):
        """Initialize shuffling of the data across epochs"""
        self._shuffled_ids = np.empty((N, epochs), dtype=np.uint64)
        for epoch in range(epochs):
            if shuffle:
                self._shuffled_ids[:, epoch] = np.random.permutation(N)
            else:
                self._shuffled_ids[:, epoch] = np.arange(N, dtype=np.uint64)


    def _generate_batch(self, x, y, epoch, batch, batch_size):
        """Generate data for one batch"""
        N = x.shape[0]
        a = batch*batch_size
        b = min(N, (batch+1)*batch_size)
        ix = self._shuffled_ids[a:b, epoch]
        return x[ix, ...], y[ix, ...]


    def _ensure_is_fit(self):
        """Raises a RuntimeError if model has not yet been fit."""
        if not self.is_fit:
            raise RuntimeError('model must first be fit')


    def sample_posterior(self, params=None, num_samples=1000):
        """Draw samples from parameter posteriors.

        TODO: Docs... params is a list of strings of params to plot

        .. admonition:: Model must be fit first!

            Before calling :meth:`.posterior` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        params : |ndarray|
            Independent variable values of the dataset to fit (aka the 
            "features").

        Returns
        -------
        dict
            Samples from the parameter posterior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarray|s with the posterior samples.  The |ndarray|s are of size
            (``num_samples``,param.shape).
        """

        # Check model has been fit
        self._ensure_is_fit()

        # Ensure num_samples is int
        if type(num_samples) is not int:
            raise TypeError('num_samples must be an int')

        # Get all params if not specified
        if params is None:
            params = [param.name for param in self._parameters]

        # Make list if string was passed
        if type(params) is str:
            params = [params]

        # Get the posterior distributions
        posteriors = dict()
        for param in self._parameters:
            if param.name in params:
                posteriors[param.name] = \
                    param.sample_posterior(num_samples=num_samples)

        return posteriors


    def plot_posterior(self, params=None):
        """Plot posterior distributions of the model's parameters.

        TODO: Docs... params is a list of strings of params to plot

        .. admonition:: Model must be fit first!

            Before calling :meth:`.plot_posteriors` on a |Model|, you must
            first :meth:`.fit` it to some data.

        """
        #TODO
        pass


    def predictive_distribution(self, x, num_samples=1000):
        """Draw samples from the model given x.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predictive_distribution` on a |Model|, you
            must first :meth:`.fit` it to some data.

        Returns array of shape (x.shape[0],y.shape[1],...,y.shape[-1],num_samples)

        """

        # Check model has been fit
        self._ensure_is_fit()

        # TODO: check x is correct shape (matches self._x_ph)

        # Draw samples from the predictive distribution
        return self._session.run(
            self.built_obj.sample(num_samples),
            feed_dict={self._x_ph: x,
                       self._batch_size_ph: [x.shape[0]]})


    def predict(self, x=None):
        """Predict dependent variable for samples in x.s

        TODO: explain how predictions are generated using the mean of each
        variational distribution

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predict` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values of data on which to
            evalutate the model.  First dimension should be equal to the number
            of samples.
            If |None|, will use the data the model was trained on (the default).

        Returns
        -------
        |ndarray|
            Predicted y-value for each sample in ``x``.  Will be of
            size ``(N, D_out)``, where N is the number of samples (equal
            to ``x.shape[0]``) and D_out is the number of output
            dimensions (equal to ``y.shape[1:]``).

        Examples
        --------
        TODO: Docs...

        """

        # Check model has been fit
        self._ensure_is_fit()

        # Use training data if none passed
        if x is None:
            x = self._x_train

        # Predict using the mean model
        #TODO: will want to use mode, not mean, for discrete distributions?
        return self._session.run(
            self.mean_obj.mean(), 
            feed_dict={self._x_ph: x,
                       self._batch_size_ph: [x.shape[0]]})


    def metrics(self, x=None, y=None, metric_list=[]):
        """Compute metrics of model performance.

        TODO: docs

        TODO: methods which just call this w/ a specific metric? for shorthand

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values of data on which to
            evalutate the model.  First dimension should be equal to the number
            of samples.
            If |None|, will use the data the model was trained on (the default).
        y : |None| or |ndarray|
            Input array of dependent variable values of data on which to
            evalutate the model.  First dimension should be equal to the number
            of samples.
            If |None|, will use the data the model was trained on (the default).
        metric_list : str or list of str
            Metrics to evaluate each epoch.  To evaluate multiple metrics, 
            pass a list of strings, where each string is a different metric.
            Available metrics:

            * 'acc': accuracy
            * 'accuracy': accuracy
            * 'mse': mean squared error
            * 'sse': sum of squared errors
            * 'mae': mean absolute error

            Default = empty list
        """

        # Make list if metric_list is not
        if isinstance(metric_list, str):
            metric_list = [metric_list]

        # Use training data if none passed
        if x is None and y is None:
            x = self._x_train
            y = self._y_train

        # Make predictions
        y_pred = self.predict(x)

        # Dict to store metrics
        metrics = dict()

        # Compute accuracy
        if 'acc' in metric_list or 'accuracy' in metric_list:
            metrics['accuracy'] = np.mean(y == y_pred)

        # Compute mean squared error
        if 'mse' in metric_list:
            metrics['mse'] = np.mean(np.square(y-y_pred))

        # Compute sum of squared errors
        if 'sse' in metric_list:
            metrics['sse'] = np.sum(np.square(y-y_pred))

        # Compute mean squared error
        if 'mae' in metric_list:
            metrics['mae'] = np.mean(y-y_pred)

        return metrics


    def plot_by(self, x, data, bins=100, what='mean'):
        """Compute and plot mean of data as a function of x.

        x should be (N,1) or (N,2)
        what can be mean, median, or count
            if mean, plot the mean of data for each bin
            etc

        returns px, py
        px is (Nbins,1) or (Nbins*Nbins,2) w/ bin centers
        py is mean of data in each bin (or count or whatevs)

        plots 2d plot w/ colormap where goes to black w/ less datapoints
        """
        #TODO
        pass


    def log_prob(self, x, y, individually=True, dist=False, num_samples=1000):
        """Compute the log probability of `y` given `x` and the model.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.log_prob` on a |Model|, you must first
            :meth:`.fit` it to some data.

        if individually is True, returns prob for each sample individually
            so return shape is (x.shape[0],?)
        if individually is False, returns product of each individual prob
            so return shape is (1,?)
        if dist is True, returns log probability posterior distribution
            (distribution of probs for lots of samples from the model)
            so return shape is (?,num_samples)
        if dist is False, returns log posterior prob assuming each parameter
            takes the mean value of its variational distribution
            so return shape iss (?,1)

        """

        # Check model has been fit
        self._ensure_is_fit()

        # TODO: make dataset/iterator/feed_dict of x and y

        # Compute probability of y given x
        # TODO
        # if dist:
        #   #sample prob from self.mean_obj.prob()?
        # else:
        #   tf_prob = self.mean_obj.prob()
        # with tf.Session() as sess:
        #   prob = sess.run(tf_prob, feed_dict=???) #make the feed dict x and y

        # TODO: but will have to SAMPLE from model and compute prob multiple times?
        # then what - take average? or median.
        # that doesn't make sense...
        # somehow need to be able to take the mean of every variational parameter...
        # which is - sigh - intercepting like in edward.
        # the predict() method should use that too
        # maybe you should also have meanify(), and meanify_args()
        #   as the equivalents of build(), and build_args(), but using the
        #   mean of any distribution?


    def log_prob_by(self, x, y, x_by, bins=100, plot=True):
        """Plot the log probability of observations `y` given `x` and the model
        as a function of independent variable(s) `x_by`.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.log_prob_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        """

        # TODO: check types of x, y, and x_by.
        # x + y should be able to be numpy arrays or ints or floats or pandas arrays
        # x_by should be able to be all those OR list of strings
        #    (length 1 or 2, col names to use if x is a pandas df)

        # Compute the model posterior probability for each observation
        probs = self.log_prob(x, y)

        # Plot probability as a fn of x_by cols of x
        px, py = self.plot_by(x[:, x_by], probs,
                              bins=bins, plot=plot)

        return px, py


    def prob(self, x, y, individually=True, dist=False, num_samples=1000):
        """Compute the probability of `y` given `x` and the model.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.prob` on a |Model|, you must first
            :meth:`.fit` it to some data.

        also, this should probably use log_prob, above, then exp it...
        """

        # TODO: evaluate log_prob w/ tf like in log_prob above
        pass


    def prob_by(self, x, y, x_by, bins=100, plot=True):
        """Plot the probability of observations `y` given `x` and the model
        as a function of independent variable(s) `x_by`.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.prob_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        """

        # TODO: same idea as log_prob_by above
        pass


    def cdf(self, x, y):
        """Compute the cumulative probability of `y` given `x` and the model.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.cdf` on a |Model|, you must first
            :meth:`.fit` it to some data.

        """

        # TODO: same idea as log_prob above
        pass


    def cdf_by(self, x, y, x_by, bins=100):
        """Plot the cumulative probability of observations `y` given `x` and
        the model as a function of independent variable(s) `x_by`.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.cdf_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        """

        # TODO: same idea as log_prob_by above
        pass


    def log_cdf(self, x, y):
        """Compute the log cumulative probability of `y` given `x` and the model.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.log_cdf` on a |Model|, you must first
            :meth:`.fit` it to some data.

        """

        # TODO: same idea as log_prob above
        pass


    def log_cdf_by(self, x, y, x_by, bins=100):
        """Plot the log cumulative probability of observations `y` given `x`
        and the model as a function of independent variable(s) `x_by`.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.log_cdf_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        """

        # TODO: same idea as log_prob_by above
        pass





class ContinuousDistribution(BaseDistribution):
    """Abstract continuous model class (used as implementation base)

    TODO: More info...

    Does this only work in class docs [2]_

    .. [2] Andrew Gelman, Ben Goodrich, Jonah Gabry, & Aki Vehtari.
        R-squared for Bayesian regression models.
        *The American Statistician*, 2018.
        https://doi.org/10.1080/00031305.2018.1549100
    """


    def predictive_prc(self, x=None, y=None):
        """Compute the percentile of each observation along the posterior
        predictive distribution.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predictive_prc` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values.  Should be
            of shape (N,D), where N is the number of samples and D
            is the number of dimensions of the independent variable.
            If |None|, will use the data the model was trained on (the default).
        y : |None| or |ndarray|
            Array of dependent variable values.  Should be of shape
            (N,D_out), where N is the number of samples (equal to
            `x.shape[0]) and D_out is the number of dimensions of
            the dependent variable.
            If |None|, will use the data the model was trained on (the default).
        """

        #TODO
        pass


    def confidence_intervals(self, x=None, prcs=[2.5, 97.5], num_samples=1000):
        """Compute confidence intervals on predictions for `x`.

        TODO: docs, prcs contains percentiles of predictive_distribution to use

        .. admonition:: Model must be fit first!

            Before calling :meth:`.confidence_intervals` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : np.ndarray
            Input array of independent variable values.  Should be of shape
            (N,D), where N is the number of samples and D is the number of
            dimensions of the independent variable.
            If |None|, will use the data the model was trained on (the default).
        prcs : list of float, or np.ndarray
            Percentiles to use as bounds of the confidence interval, between 0
            and 100.
            Default = [2.5, 97.5]
        num_samples : int
            Number of samples from the posterior predictive distribution to
            take to compute the confidence intervals.
            Default = 1000

        Returns
        -------
        conf_intervals : np.ndarray
            Confidence intervals on the predictions for samples in `x`.
        """

        # Check types
        # TODO

        # Check model has been fit
        self._ensure_is_fit()

        # Compute percentiles of the predictive distribution
        pred_dist = self.predictive_distribution(x, num_samples=num_samples)
        return np.percentile(pred_dist, prcs)


    def pred_dist_covered(self, x=None, y=None, prc=95.0):
        """Compute whether each observation was covered by the
        inner `prc` percentile of the posterior predictive
        distribution.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.pred_dist_covered` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values.  Should be
            of shape (N,D), where N is the number of samples and D
            is the number of dimensions of the independent variable.
            If |None|, will use the data the model was trained on (the default).
        y : |None| or |ndarray|
            Array of dependent variable values.  Should be of shape
            (N,D_out), where N is the number of samples (equal to
            `x.shape[0]) and D_out is the number of dimensions of
            the dependent variable.
            If |None|, will use the data the model was trained on (the default).
        """

        # Check model has been fit
        self._ensure_is_fit()

        #TODO
        pass


    def pred_dist_coverage(self, x=None, y=None, prc=95.0):
        """Compute the coverage of the inner `prc` percentile of the
        posterior predictive distribution.

        TODO: Docs...
        returns a scalar (from 0 to 100)

        .. admonition:: Model must be fit first!

            Before calling :meth:`.pred_dist_coverage` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values.  Should be
            of shape (N,D), where N is the number of samples and D
            is the number of dimensions of the independent variable.
            If |None|, will use the data the model was trained on (the default).
        y : |None| or |ndarray|
            Array of dependent variable values.  Should be of shape
            (N,D_out), where N is the number of samples (equal to
            `x.shape[0]) and D_out is the number of dimensions of
            the dependent variable.
            If |None|, will use the data the model was trained on (the default).
        """

        # Check model has been fit
        self._ensure_is_fit()

        #TODO
        pass


    def coverage_by(self, x_by, x=None, y=None, prc=95.0, bins=100, plot=True):
        """Compute and plot the coverage of the inner `prc`
        percentile of the posterior predictive distribution as a
        function of specified independent variables.

        TODO: Docs...
        x_by should be int or length-2 list of ints which specifies what column of x to plot by
        returns x and coverage matrix

        .. admonition:: Model must be fit first!

            Before calling :meth:`.coverage_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values.  Should be
            of shape (N,D), where N is the number of samples and D
            is the number of dimensions of the independent variable.
            If |None|, will use the data the model was trained on (the default).
        y : |None| or |ndarray|
            Array of dependent variable values.  Should be of shape
            (N,D_out), where N is the number of samples (equal to
            `x.shape[0]) and D_out is the number of dimensions of
            the dependent variable.
            If |None|, will use the data the model was trained on (the default).
        """

        # Compute whether each sample was covered by the interval
        covered = self.pred_dist_covered(x, y, prc)

        # TODO: alternatively, x_by should be able to be any array_like
        # as long as it's same size as x.shape[0]

        # Plot probability as a fn of x_by cols of x
        px, py = self.plot_by(x[:, x_by], covered,
                              bins=bins, plot=plot)

        return px, py


    def calibration_curve(self, x=None, y=None, split_by=None,
                          bins=10, plot=False):
        """Plot and/or return calibration curve.

        Plots and returns the calibration curve (the percentile of the posterior
        predictive distribution on the x-axis, and the percent of samples which
        actually fall into that range on the y-axis).

        .. admonition:: Model must be fit first!

            Before calling
            :meth:`calibration_curve() <.ContinuousModel.calibration_curve>` on
            a |Model|, you must first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values.  Should be
            of shape (N,D), where N is the number of samples and D
            is the number of dimensions of the independent variable.
            If |None|, will use the data the model was trained on (the default).
        y : |None| or |ndarray|
            Array of dependent variable values.  Should be of shape
            (N,D_out), where N is the number of samples (equal to
            `x.shape[0]) and D_out is the number of dimensions of
            the dependent variable.
            If |None|, will use the data the model was trained on (the default).
        split_by : int
            Draw the calibration curve independently for datapoints
            with each unique value in `x[:,split_by]` (a categorical
            column).
        bins : int, list of float, or |ndarray|
            Bins used to compute the curve.  If an integer, will use
            `bins` evenly-spaced bins from 0 to 1.  If a vector,
            `bins` is the vector of bin edges.

        Returns
        -------
        cx : |ndarray|
            Vector of percentiles (the middle of each percentile
            bin).  Length is determined by `bins`.
        cy : |ndarray|
            Vector of percentages of samples which fell within each
            percentile bin of the posterior predictive distribution.

        See Also
        --------
        predictive_distribution : used to generate the posterior
            predictive distribution.

        Notes
        -----
        TODO: Docs...

        Examples
        --------
        TODO: Docs...

        """

        # Check model has been fit
        self._ensure_is_fit()

        #TODO
        pass


    def r_squared(self, x=None, y=None, num_samples=1000, plot=False):
        """Compute the Bayesian R-squared value.

        Compute the Bayesian R-squared distribution :ref:`[1] <ref_r_squared>`.
        TODO: more info and docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.r_squared` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values.  If |None|, will use
            the data the model was trained on (the default).
        y : |None| or |ndarray|
            Array of dependent variable values.  If |None|, will use
            the data the model was trained on (the default).
        num_samples : int
            Number of posterior draws to use for computing the r-squared
            distribution.  Default = `1000`.
        plot : bool
            Whether to plot the r-squared distribution

        Returns
        -------
        |ndarray|
            Samples from the r-squared distribution.  Size: ``(num_samples,)``.

        Notes
        -----
        TODO: Docs...

        Examples
        --------
        TODO: Docs...

        References
        ----------
        .. _ref_r_squared:
        .. [1] Andrew Gelman, Ben Goodrich, Jonah Gabry, & Aki Vehtari.
            R-squared for Bayesian regression models.
            *The American Statistician*, 2018.
            https://doi.org/10.1080/00031305.2018.1549100
        """
        #TODO
        pass


    def residuals(self, x=None, y=None, plot=False):
        """Compute the residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values.  Should be
            of shape (N,D), where N is the number of samples and D
            is the number of dimensions of the independent variable.
            If |None|, will use the data the model was trained on (the default).
        y : |None| or |ndarray|
            Array of dependent variable values.  Should be of shape
            (N,D_out), where N is the number of samples (equal to
            `x.shape[0]) and D_out is the number of dimensions of
            the dependent variable.
            If |None|, will use the data the model was trained on (the default).

        """
        # TODO
        pass



class DiscreteDistribution(BaseDistribution):
    """Abstract categorical model class (used as implementation base)

    TODO: More info...

    """

    def predict(self, x=None):
        """Predict dependent variable for samples in x.s

        TODO: explain how predictions are generated using the MODE of each
        variational distribution

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predict` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |None| or |ndarray|
            Input array of independent variable values of data on which to
            evalutate the model.  First dimension should be equal to the number
            of samples.
            If |None|, will use the data the model was trained on (the default).

        Returns
        -------
        |ndarray|
            Predicted y-value for each sample in ``x``.  Will be of
            size ``(N, D_out)``, where N is the number of samples (equal
            to ``x.shape[0]``) and D_out is the number of output
            dimensions (equal to ``y.shape[1:]``).

        Examples
        --------
        TODO: Docs...

        """

        # Check model has been fit
        self._ensure_is_fit()

        # Use training data if none passed
        if x is None:
            x = self._x_train

        # Predict using the mode of the mean model
        return self._session.run(
            self.mean_obj.mode(), 
            feed_dict={self._x_ph: x,
                       self._batch_size_ph: [x.shape[0]]})


    def calibration_curve(self, x, y, split_by=None, bins=10):
        """Plot and return calibration curve.

        Plots and returns the calibration curve (estimated
        probability of outcome vs the true probability of that
        outcome).

        .. admonition:: Model must be fit first!

            Before calling
            :meth:`calibration_curve() <.CategoricalModel.calibration_curve>`
            on a |Model|, you must first :meth:`.fit` it to some data.

        Parameters
        ----------
        x:
        y:
        split_by: draw curve independently for datapoints with
            each unique value in this categorical column number.
        bins: bins used to compute the curve.  An integer to
            specify the number of evenly-spaced bins from 0 to
            1, or a list or array-like to specify the bin edges.

        #TODO: split by continuous cols as well? Then will need to define bins or edges too

        TODO: Docs...

        """

        # Check model has been fit
        self._ensure_is_fit()

        #TODO
        pass


    # TODO: are there categorical equivalents of predictive_prc,
    # pred_dist_covered, pred_dist_coverage, and coverage_by?

    # TODO: confusion_matrix (plot/return the confusion matrix of predictions)

