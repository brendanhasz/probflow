"""Abstract classes.

TODO: more info...

----------

"""

__all__ = [
    'REQUIRED',
    'BaseObject',
    'BaseParameter',
    'BaseLayer',
    'BaseDistribution',
    'ContinuousDistribution',
    'DiscreteDistribution',
]

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from .utils.data import process_data, process_xy_data, test_train_split
from .utils.plotting import plot_line, fill_between


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


    def _validate_kwargs(self, kwargs):
        """Ensure the keyword arguments have correct types, etc."""
        pass



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
        self._validate_kwargs(self.kwargs)

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


    def fit(self, x, y, data=None,
            dtype=tf.float32,
            batch_size=128,
            epochs=100,
            optimizer='adam',
            learning_rate=0.01,
            metrics=[],
            verbose=True,
            validation_split=0.0,
            validation_shuffle=True,
            shuffle=True,
            record=None,
            record_freq='batch'):
        """Fit model.

        TODO: Docs...

        TODO: brief math about variational inference :ref:`[1] <ref_bbb>`
        (just to say that the loss used is -ELBO = KL - log_likelihood)

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables.
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as dependent variables.
        data : |None| or |DataFrame|
            Data for the fit.  If ``data`` is |None|, :meth:`.fit` assumes 
            ``x`` and ``y`` are |ndarray|s.  If ``data`` is a |DataFrame|,
            :meth:`.fit` assumes ``x`` and ``y`` are strings or lists of 
            strings containing the columns from ``data`` to use.
        dtype : |DType|
            Cast the input data to this type, and use this type for parameters.
        batch_size : int or None
            Number of samples per training batch.  Use None to use all samples
            per batch.  Default = 128
        epochs : int
            Number of epochs to train for (1 epoch is one full cycle through
            all the training data).  Default = 100
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
            Whether to shuffle which data is used for validation.  If False,
            the last ``validation_split`` proportion of the input data is used
            for validation.
            Default = True
        shuffle : bool
            Whether to shuffle the training data before each trainin epoch.
            Default = True
        record : None or str or list of str
            Parameters to record over the course of training.  If ``record`` is
            None, no parameter recording occurrs.  If ``record`` is ``'all'``,
            all parameters are recorded.  If ``record`` is a string containing
            the name of a |Parameter|, that parameter's variational posterior
            parameters are recorded.  If ``record`` is a list, each element of
            the list should be a string with the name of a |Parameter| to 
            record.
        record_freq : string {'batch' or 'epoch'}
            Recording frequency.  If ``record_freq`` is ``'batch'``, 
            variational posterior parameters will be recorded once per batch.
            If ``record_freq`` is ``'epoch'``, variational posterior parameters
            will only be recorded once per epoch (which saves memory if your
            model has many parameters).


        References
        ----------
        .. _ref_bbb:
        .. [1] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and 
            Daan Wierstra. Weight uncertainty in neural networks. 
            *arXiv preprint*, 2015. http://arxiv.org/abs/1505.05424


        """


        def make_placeholders(x, y, dtype):
            """Create x, y, and batch_shape placeholders"""

            # Store pointer to training data
            self._train = dict()
            self._train['x'] = x
            self._train['y'] = y

            # Data placeholders
            x_shape = list(x_train.shape)
            y_shape = list(y_train.shape)
            x_shape[0] = None
            y_shape[0] = None
            x_data = tf.placeholder(dtype, x_shape)
            y_data = tf.placeholder(dtype, y_shape)

            # Batch shape
            batch_size_ph = tf.placeholder(tf.int32, [1])

            # Store placeholders
            self._ph = dict()
            self._ph['batch_size'] = batch_size_ph
            self._ph['x'] = x_data
            self._ph['y'] = y_data

            return x_data, y_data, batch_size_ph


        def initialize_shuffles(N, epochs, shuffle):
            """Initialize shuffling of the data across epochs"""
            self._shuffled_ids = np.empty((N, epochs), dtype=np.uint64)
            for epoch in range(epochs):
                if shuffle:
                    self._shuffled_ids[:, epoch] = np.random.permutation(N)
                else:
                    self._shuffled_ids[:, epoch] = \
                        np.arange(N, dtype=np.uint64)


        def generate_batch(x, y, epoch, batch, batch_size):
            """Generate data for one batch"""
            N = x.shape[0]
            a = batch*batch_size
            b = min(N, (batch+1)*batch_size)
            ix = self._shuffled_ids[a:b, epoch]
            return x[ix, ...], y[ix, ...], [ix.shape[0]]


        def init_records(to_record, record_freq, epochs, n_batch):
            """Initialize dicts and arrays for recording posterior params"""
            if record_freq == 'batch':
                Nrecords = int(epochs*n_batch)
            else:
                Nrecords = int(epochs)
            records = dict()
            records['x_epochs'] = np.linspace(1, epochs, Nrecords)
            for record in to_record:
                if record not in [p.name for p in self._parameters]:
                    raise ValueError(record+' is not a parameter')
            for param in self._parameters:
                if param.name in to_record:
                    records[param.name] = dict()
                    t_shape = [Nrecords] + param.shape
                    for post_arg in param._params:
                        records[param.name][post_arg] = \
                            np.full(t_shape, np.nan)
            return records

        def save_records(ix):
            """Save posterior parameter values"""
            for param in self._parameters:
                if param.name in self._records:
                    for post_arg in param._params:
                        self._records[param.name][post_arg][ix,...] = \
                            self._session.run(param._params[post_arg])


        # Check input types
        if not isinstance(dtype, tf.DType):
            raise TypeError('dtype must be a TensorFlow DType')
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be an int')
        if batch_size < 1:
            raise TypeError('batch_size must be greater than 0')
        if not isinstance(epochs, int):
            raise TypeError('epochs must be an int')
        if epochs < 0:
            raise TypeError('epochs must be non-negative')
        if not isinstance(optimizer, str):
            raise TypeError('optimizer must be a string')
        if optimizer not in ['adam']:
            raise TypeError('optimizer must be one of: \'adam\'')
        if not isinstance(learning_rate, float):
            raise TypeError('learning_rate must be a float')
        if learning_rate < 0:
            raise TypeError('learning_rate must be non-negative')
        if not isinstance(verbose, bool):
            raise TypeError('verbose must be True or False')
        if not isinstance(validation_split, float):
            raise TypeError('validation_split must be a float')
        if validation_split < 0 or validation_split > 1:
            raise TypeError('validation_split must be between 0 and 1')
        if not isinstance(validation_shuffle, bool):
            raise TypeError('validation_shuffle must be True or False')
        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be True or False')
        if record is not None and not isinstance(record, (str, list)):
            raise TypeError('record must be None, a string, or a list')
        if isinstance(record, list):
            if not all([isinstance(e, str) for e in record]):
                raise TypeError('record must be a list of strings')
        if not isinstance(record_freq, str):
            raise TypeError('record_freq must be a string')
        if record_freq not in ['batch', 'epoch']:
            raise ValueError('record_freq must be \'batch\' or \'epoch\'')

        # Process the input data
        x, y = process_xy_data(self, x, y, data)

        # TODO: how to support integer columns?

        # Split data into training and validation data
        N, x_train, y_train, x_val, y_val = \
            test_train_split(x, y, validation_split, validation_shuffle)

        # Create placeholders for input data
        x_data, y_data, batch_size_ph = \
            make_placeholders(x_train, y_train, dtype)

        # Initialize the shuffling of training data
        initialize_shuffles(N, epochs, shuffle)

        # Recursively build this model and its args
        self.build(x_data, batch_size_ph)

        # Set up TensorFlow graph for per-sample losses
        self.log_loss = (self.samp_loss_sum +  #size (batch_size,)
                         self._log_loss(y_data))
        self.mean_log_loss = (self.mean_loss_sum + #size (batch_size,)
                              self._mean_log_loss(y_data))
        self.kl_loss = tf.cast(self.kl_loss_sum + self._kl_loss(), dtype)

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

        # Ensure this model contains parameters
        if len(self._parameters) == 0:
            raise RuntimeError('model contains no parameters, cannot fit it!')

        # Create the TensorFlow session and assign it to each parameter
        self._session = tf.Session()
        for param in self._parameters:
            param._session = self._session

        # Initializers
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self._session.run(init_op)

        # Set up arrays for recording
        n_batch = int(np.ceil(N/batch_size)) #number of batches per epoch
        if isinstance(record, str):
            if record == 'all':
                record = [p.name for p in self._parameters]
            else:
                record = [record] #make list if not
        if record is not None:
            self._records = init_records(record, record_freq, epochs, n_batch)

        # Fit the model
        self.is_fit = True
        print_batches = int(np.ceil(n_batch/10)) #info each print_batches batch
        for epoch in range(epochs):

            # Print progress
            if verbose:
                print('Epoch %d / %d' % (epoch, epochs))

            # Train on each batch in this epoch
            for batch in range(n_batch):
                b_x, b_y, b_n = generate_batch(x_train, y_train, 
                                               epoch, batch, batch_size)
                self._session.run(train_op,
                                  feed_dict={x_data: b_x,
                                             y_data: b_y,
                                             batch_size_ph: b_n})

                # Record variational posteriors each batch
                if record is not None and record_freq == 'batch':
                    save_records(epoch*n_batch + batch)

                # Print progress
                if verbose and batch % print_batches == 0:
                    print("  Batch %d / %d (%0.1f%%)\r" %
                          (batch+1, n_batch, 100.0*batch/n_batch), end='')

            # Record variational posteriors each epoch
            if record is not None and record_freq == 'epoch':
                save_records(epoch)

            # Evaluate metrics
            print(60*' '+"\r", end='')
            if metrics:
                md = self.metrics(x_val, y_val, metrics)
                print('  '+(4*' ').join([m+': '+str(md[m]) for m in md]))

        # Finished!
        print('Done!')


    def _ensure_is_fit(self):
        """Raises a RuntimeError if model has not yet been fit."""
        if not self.is_fit:
            raise RuntimeError('model must first be fit')


    def predictive_distribution(self, x=None, data=None, num_samples=1000):
        """Draw samples from the model given x.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predictive_distribution` on a |Model|, you
            must first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            Data for the fit.  If ``data`` is |None|, :meth:`.fit` assumes 
            ``x`` and ``y`` are |ndarray|s.  If ``data`` is a |DataFrame|,
            :meth:`.fit` assumes ``x`` and ``y`` are strings or lists of 
            strings containing the columns from ``data`` to use.
        num_samples : int
            TODO

        Returns array of shape 
        (x.shape[0],y.shape[1],...,y.shape[-1],num_samples)

        """

        # Check inputs

        # Check model has been fit
        self._ensure_is_fit()

        # Process input data
        x = process_data(self, x, data)

        # Draw samples from the predictive distribution
        return self._session.run(
            self.built_obj.sample(num_samples),
            feed_dict={self._ph['x']: x,
                       self._ph['batch_size']: [x.shape[0]]})


    # TODO: plot_predictive_distribution()


    def predict(self, x=None, data=None):
        """Predict dependent variable for samples in x.s

        TODO: explain how predictions are generated using the mean of each
        variational distribution

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predict` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            Data for the fit.  If ``data`` is |None|, :meth:`.fit` assumes 
            ``x`` and ``y`` are |ndarray|s.  If ``data`` is a |DataFrame|,
            :meth:`.fit` assumes ``x`` and ``y`` are strings or lists of 
            strings containing the columns from ``data`` to use.

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

        # Process input data
        x = process_data(self, x, data)

        # Predict using the mean model
        return self._session.run(
            self.mean_obj.mean(), 
            feed_dict={self._ph['x']: x,
                       self._ph['batch_size']: [x.shape[0]]})


    def metrics(self, x=None, y=None, data=None, metric_list=[]):
        """Compute metrics of model performance.

        TODO: docs

        TODO: methods which just call this w/ a specific metric? for shorthand

        .. admonition:: Model must be fit first!

            Before calling :meth:`.metrics` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as dependent variables.  If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            Data for the fit.  If ``data`` is |None|, :meth:`.fit` assumes 
            ``x`` and ``y`` are |ndarray|s.  If ``data`` is a |DataFrame|,
            :meth:`.fit` assumes ``x`` and ``y`` are strings or lists of 
            strings containing the columns from ``data`` to use.
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

        # Check types
        if not isinstance(metric_list, (str, list)):
            raise ValueError('metric_list must be a string or list of strings')
        if isinstance(metric_list, list):
            if not all([isinstance(e, str) for e in metric_list]):
                raise ValueError('metric_list must be a list of strings')

        # Make list if metric_list is not
        if isinstance(metric_list, str):
            metric_list = [metric_list]

        # Process input data
        x, y = process_data(self, x, y, data)

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

        # TODO: cross-entropy, etc

        return metrics


    def posterior_mean(self, params=None):
        """Get the mean of the posterior distribution(s).

        TODO: Docs... params is a list of strings of params to plot

        .. admonition:: Model must be fit first!

            Before calling :meth:`.posterior_mean` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        params : list
            List of parameter names to sample.  Each element should be a str.

        Returns
        -------
        dict
            Means of the parameter posterior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarray|s with the posterior means.  The |ndarray|s are the same
            size as each parameter.
        """

        # Check model has been fit
        self._ensure_is_fit()

        # Check parameter list
        param_dict = self._validate_params(params)

        # Get the posterior means
        posterior_means = dict()
        for name, param in param_dict.items():
            posterior_means[name] = param.posterior_mean()

        return posterior_means


    def sample_posterior(self, params=None, num_samples=1000):
        """Draw samples from parameter posteriors.

        TODO: Docs... params is a list of strings of params to plot

        .. admonition:: Model must be fit first!

            Before calling :meth:`.sample_posterior` on a |Model|, you must 
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        params : list
            List of parameter names to sample.  Each element should be a str.
        num_samples : int
            Number of samples to take from each posterior distribution.
            Default = 1000

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

        # Check parameter list
        param_dict = self._validate_params(params)

        # Check other inputs
        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an int')
        if num_samples < 1:
            raise ValueError('num_samples must be greater than 0')

        # Get the posterior distributions
        posteriors = dict()
        for name, param in param_dict.items():
            posteriors[name] = param.sample_posterior(num_samples=num_samples)

        return posteriors


    def plot_posterior(self,
                       params=None,
                       num_samples=1000,
                       style='fill',
                       cols=1,
                       bins=20,
                       ci=0.0,
                       bw=0.075,
                       color=None,
                       alpha=0.4):
        """Plot posterior distributions of the model's parameters.

        TODO: Docs... params is a list of strings of params to plot

        .. admonition:: Model must be fit first!

            Before calling :meth:`.plot_posterior` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        params : str or list
            List of parameters to plot.  Default is to plot the posterior of
            all parameters in the model.
        num_samples : int
            Number of samples to take from each posterior distribution for
            estimating the density.  Default = 1000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns.
        bins : int or list or |ndarray|
            Number of bins to use for the posterior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram of the density
        """

        # Check model has been fit
        self._ensure_is_fit()

        # Check parameter list
        param_dict = self._validate_params(params)

        # Check other inputs
        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an int')
        if num_samples < 1:
            raise ValueError('num_samples must be greater than 0')
        if type(style) is not str or style not in ['fill', 'line', 'hist']:
            raise TypeError("style must be \'fill\', \'line\', or \'hist\'")
        if type(cols) is not int:
            raise TypeError('cols must be an integer')
        if not isinstance(bins, (int, float, np.ndarray)):
            raise TypeError('bins must be an int or list or numpy vector')
        if type(ci) is not float or ci<0.0 or ci>1.0:
            raise TypeError('ci must be a float between 0 and 1')
        if type(alpha) is not float or alpha<0.0 or alpha>1.0:
            raise TypeError('alpha must be a float between 0 and 1')

        # Plot each parameter's posterior distributions in separate subplot
        rows = np.ceil(len(param_dict)/cols)
        for ix, param in enumerate(param_dict):
            plt.subplot(rows, cols, ix+1)
            param_dict[param].plot_posterior(num_samples=num_samples, 
                                             style=style, bins=bins, ci=ci,
                                             color=color)


    def sample_prior(self, params=None, num_samples=10000):
        """Draw samples from parameter priors.

        TODO: Docs... params is a list of strings of params to plot

        .. admonition:: Model must be fit first!

            Before calling :meth:`.sample_prior` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        params : list
            List of parameter names to sample.  Each element should be a str.
        num_samples : int
            Number of samples to take from each prior distribution.
            Default = 10000

        Returns
        -------
        dict
            Samples from the parameter prior distributions.  A dictionary
            where the keys contain the parameter names and the values contain
            |ndarray|s with the prior samples.  The |ndarray|s are of size
            (``num_samples``,param.shape).
        """

        # Check model has been fit
        self._ensure_is_fit()

        # Check parameter list
        param_dict = self._validate_params(params)

        # Check other inputs
        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an int')
        if num_samples < 1:
            raise ValueError('num_samples must be greater than 0')

        # Get the prior distribution samples
        priors = dict()
        for name, param in param_dict.items():
            priors[name] = param.sample_prior(num_samples=num_samples)

        return priors


    def plot_prior(self,
                   params=None,
                   num_samples=10000,
                   style='fill',
                   cols=1,
                   bins=20,
                   ci=0.0,
                   bw=0.075,
                   color=None,
                   alpha=0.4):
        """Plot prior distributions of the model's parameters.

        TODO: Docs... params is a list of strings of params to plot

        .. admonition:: Model must be fit first!

            Before calling :meth:`.plot_prior` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        params : |None| or str or list of str
            List of parameters to plot.  Default is to plot the prior of
            all parameters in the model.
        num_samples : int
            Number of samples to take from each prior distribution.
            Default = 10000
        style : str
            Which style of plot to show.  Available types are:

            * ``'fill'`` - filled density plot (the default)
            * ``'line'`` - line density plot
            * ``'hist'`` - histogram

        cols : int
            Divide the subplots into a grid with this many columns.
        bins : int or list or |ndarray|
            Number of bins to use for the prior density histogram (if 
            ``style='hist'``), or a list or vector of bin edges.
        ci : float between 0 and 1
            Confidence interval to plot.  Default = 0.0 (i.e., not plotted)
        bw : float
            Bandwidth of the kernel density estimate (if using ``style='line'``
            or ``style='fill'``).  Default is 0.075
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        alpha : float between 0 and 1
            Transparency of fill/histogram of the density
        """

        # Check model has been fit
        self._ensure_is_fit()

        # Check parameter list
        param_dict = self._validate_params(params)

        # Check other inputs
        if not isinstance(num_samples, int):
            raise TypeError('num_samples must be an int')
        if num_samples < 1:
            raise ValueError('num_samples must be greater than 0')
        if type(style) is not str or style not in ['fill', 'line', 'hist']:
            raise TypeError("style must be \'fill\', \'line\', or \'hist\'")
        if type(cols) is not int:
            raise TypeError('cols must be an integer')
        if not isinstance(bins, (int, float, np.ndarray)):
            raise TypeError('bins must be an int or list or numpy vector')
        if type(ci) is not float or ci<0.0 or ci>1.0:
            raise TypeError('ci must be a float between 0 and 1')
        if type(alpha) is not float or alpha<0.0 or alpha>1.0:
            raise TypeError('alpha must be a float between 0 and 1')

        # Plot each parameter's prior distribution in separate subplot
        rows = np.ceil(len(param_dict)/cols)
        for ix, param in enumerate(param_dict):
            plt.subplot(rows, cols, ix+1)
            param_dict[param].plot_prior(num_samples=num_samples, style=style, 
                                         bins=bins, ci=ci, color=color)


    def plot_posterior_over_training(self, 
                                     params=None,
                                     cols=1,
                                     ci=[0.1, 0.5, 0.95],
                                     marker='-',
                                     alpha=0.3,
                                     color=None):
        """Plot the variational posteriors over the course of training.

        Plots confidence intervals of the variational posterior distributions
        across training epochs.

        TODO: more docs... 

        Parameters
        ----------
        params : |None| or str or list of str
            List of parameters to plot.  Default is to plot the posteriors of
            all parameters in the model over the course of training.
        cols : int
            Divide the subplots into a grid with this many columns.
        ci : list of float between 0 and 1
            Confidence intervals to plot.  Default = ``[0.1, 0.5, 0.95]``.
        marker : str or matplotlib linespec
            Line marker to use.
        alpha : float between 0 and 1
            Transparency of density polygons
        color : matplotlib color code or list of them
            Color(s) to use to plot the distribution.
            See https://matplotlib.org/tutorials/colors/colors.html
            Default = use the default matplotlib color cycle
        """

        # Check model has been fit
        self._ensure_is_fit()

        # Check parameter list
        param_dict = self._validate_params(params, rec=True)

        # Check other inputs
        if not isinstance(cols, int):
            raise TypeError('cols must be an integer')
        if cols < 1:
            raise ValueError('cols must be greater than 0')
        if not isinstance(ci, (float, list)):
            raise TypeError('ci must be a float or a list of floats')
        if isinstance(ci, list):
            for c in ci:
                if not isinstance(c, float):
                    raise TypeError('ci must be a float or a list of floats')

        # X values
        x_vals = self._records['x_epochs']
        x_res = len(x_vals)

        # Compute confidence interval percentiles
        ci = np.array(ci)
        ci_lb = 0.5 - ci/2.0
        ci_ub = 0.5 + ci/2.0

        # Plot confidence intervals over training
        ix = 0
        rows = np.ceil(len(param_dict)/cols)
        for name, param in param_dict.items():

            # Create a TFP distribution of the posterior across training
            t_post = param.posterior_fn(**self._records[name])
            t_post.build(None, None)
            t_post = t_post.built_obj

            # Compute the quantiles
            tfo = param.transform
            lb = np.empty([len(ci), x_res]+param.shape)
            ub = np.empty([len(ci), x_res]+param.shape)
            with tf.Session() as sess:
                for iy in range(len(ci)):
                    lb[iy,...] = sess.run(tfo(t_post.quantile(ci_lb[iy])))
                    ub[iy,...] = sess.run(tfo(t_post.quantile(ci_ub[iy])))

            # Plot the quantiles
            plt.subplot(rows, cols, ix+1)
            fill_between(x_vals, lb, ub, xlabel='Epoch', ylabel=name, 
                         alpha=alpha, color=color)
            ix += 1


    def plot_posterior_args_over_training(self, 
                                          params=None,
                                          cols=1,
                                          marker='-'):
        """Plot the variational posterior's parameters across training.

        TODO: more docs...

        Parameters
        ----------
        params : |None| or str or list of str
            List of parameters to plot.  Default is to plot the posteriors of
            all parameters in the model over the course of training.
        cols : int
            Divide the subplots into a grid with this many columns.
        marker : str or matplotlib linespec
            Line marker to use.
        """

        # Check model has been fit
        self._ensure_is_fit()

        # Check parameter list
        param_dict = self._validate_params(params, rec=True)

        # Check other inputs
        if not isinstance(cols, int):
            raise TypeError('cols must be an integer')
        if cols < 1:
            raise ValueError('cols must be greater than 0')

        # Count how many posterior arguments there are
        n_args = 0
        for _, param in param_dict.items():
            n_args += len(param._params)

        # Plot each variational posterior's argument in separate subplot
        rows = np.ceil(n_args/cols)
        ix = 0
        x_vals = self._records['x_epochs']
        for name, param in param_dict.items():
            for arg in param._params:
                plt.subplot(rows, cols, ix+1)
                plot_line(x_vals, self._records[name][arg], fmt=marker,
                          xlabel='Epoch', ylabel=name+'\n'+arg)
                ix += 1


    def _validate_params(self, params, rec=False):
        """Check params list is valid."""

        # Check types
        if params is not None and not isinstance(params, (list, str)):
            raise TypeError('params must be None or a list of str')
        if type(params) is list:
            for param in params:
                if not isinstance(param, str):
                    raise TypeError('params must be None or a list of str')

        # Get all params if not specified
        if params is None:
            params = [p.name for p in self._parameters]

        # Make list if string was passed
        if isinstance(params, str):
            params = [params]

        # Check requested parameters are in the model
        for param in params:
            if param not in [p.name for p in self._parameters]:
                raise ValueError('Parameter \''+param+'\' not in this model')

        # Check requested parameters were recorded
        if rec:
            for param in params:
                if param not in self._records:
                    raise ValueError('Parameter \''+param+'\' was not ' +
                                     'recorded. To record, set the record ' +
                                     'argument when calling fit().')

        # Make dict of params to get
        param_dict = dict()
        for param in self._parameters:
            if param.name in params:
                param_dict[param.name] = param

        return param_dict


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


    def log_prob(self, x=None, y=None, data=None, 
                 individually=True, dist=False, num_samples=1000):
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

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            Data for the fit.  If ``data`` is |None|, :meth:`.fit` assumes 
            ``x`` and ``y`` are |ndarray|s.  If ``data`` is a |DataFrame|,
            :meth:`.fit` assumes ``x`` and ``y`` are strings or lists of 
            strings containing the columns from ``data`` to use.

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


    def log_prob_by(self, x_by, x=None, y=None, data=None, 
                    bins=100, plot=True):
        """Plot the log probability of observations `y` given `x` and the model
        as a function of independent variable(s) `x_by`.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.log_prob_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x_by : int or string or 2-element list of int or string
            Independent variable to plot the log probability as a function of.
            TODO
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            Data for the fit.  If ``data`` is |None|, :meth:`.fit` assumes 
            ``x`` and ``y`` are |ndarray|s.  If ``data`` is a |DataFrame|,
            :meth:`.fit` assumes ``x`` and ``y`` are strings or lists of 
            strings containing the columns from ``data`` to use.

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


    def prob(self, x=None, y=None, data=None,
             individually=True, dist=False, num_samples=1000):
        """Compute the probability of `y` given `x` and the model.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.prob` on a |Model|, you must first
            :meth:`.fit` it to some data.

        also, this should probably use log_prob, above, then exp it...

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.


        """

        # TODO: evaluate log_prob w/ tf like in log_prob above
        pass


    def prob_by(self, x_by, x=None, y=None, data=None, bins=100, plot=True):
        """Plot the probability of observations `y` given `x` and the model
        as a function of independent variable(s) `x_by`.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.prob_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x_by : TODO
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.

        """

        # TODO: same idea as log_prob_by above
        pass


    def cdf(self, x=None, y=None, data=None):
        """Compute the cumulative probability of `y` given `x` and the model.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.cdf` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.

        """

        # TODO: same idea as log_prob above
        pass


    def cdf_by(self, x_by, x=None, y=None, data=None, bins=100):
        """Plot the cumulative probability of observations `y` given `x` and
        the model as a function of independent variable(s) `x_by`.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.cdf_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x_by : TODO
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
        bins : TODO
        """

        # TODO: same idea as log_prob_by above
        pass


    def log_cdf(self, x=None, y=None, data=None):
        """Compute the log cumulative probability of `y` given `x` and the model.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.log_cdf` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.

        """

        # TODO: same idea as log_prob above
        pass


    def log_cdf_by(self, x_by, x=None, y=None, data=None, bins=100):
        """Plot the log cumulative probability of observations `y` given `x`
        and the model as a function of independent variable(s) `x_by`.

        TODO: docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.log_cdf_by` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x_by : TODO
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
        bins : TODO
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


    def predictive_prc(self, x=None, y=None, data=None):
        """Compute the percentile of each observation along the posterior
        predictive distribution.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predictive_prc` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
        """

        #TODO
        pass


    def confidence_intervals(self, x=None, data=None,
                             prcs=[2.5, 97.5], num_samples=1000):
        """Compute confidence intervals on predictions for `x`.

        TODO: docs, prcs contains percentiles of predictive_distribution to use

        .. admonition:: Model must be fit first!

            Before calling :meth:`.confidence_intervals` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x``.  If ``data`` is |None|, 
            it is assumed that ``x`` is a |ndarray|.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` is a string
            or list of strings containing the columns from ``data`` to use.
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


    def pred_dist_covered(self, x=None, y=None, data=None, prc=95.0):
        """Compute whether each observation was covered by the
        inner `prc` percentile of the posterior predictive
        distribution.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.pred_dist_covered` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
        """

        # Check model has been fit
        self._ensure_is_fit()

        #TODO
        pass


    def pred_dist_coverage(self, x=None, y=None, data=None, prc=95.0):
        """Compute the coverage of the inner `prc` percentile of the
        posterior predictive distribution.

        TODO: Docs...
        returns a scalar (from 0 to 100)

        .. admonition:: Model must be fit first!

            Before calling :meth:`.pred_dist_coverage` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
        """

        # Check model has been fit
        self._ensure_is_fit()

        #TODO
        pass


    def coverage_by(self, x_by, x=None, y=None, data=None, 
                    prc=95.0, bins=100, plot=True):
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
        x_by : TODO
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
        TODO: other args
        """

        # Compute whether each sample was covered by the interval
        covered = self.pred_dist_covered(x, y, prc)

        # TODO: alternatively, x_by should be able to be any array_like
        # as long as it's same size as x.shape[0]

        # Plot probability as a fn of x_by cols of x
        px, py = self.plot_by(x[:, x_by], covered,
                              bins=bins, plot=plot)

        return px, py


    def calibration_curve(self, x=None, y=None, data=None,
                          split_by=None, bins=10, plot=False):
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
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
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


    def r_squared(self, x=None, y=None, data=None, 
                  num_samples=1000, plot=False):
        """Compute the Bayesian R-squared value.

        Compute the Bayesian R-squared distribution :ref:`[1] <ref_r_squared>`.
        TODO: more info and docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.r_squared` on a |Model|, you must
            first :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
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


    def residuals(self, x=None, y=None, data=None):
        """Compute the residuals of the model's predictions.

        TODO: docs...

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.

        """
        # TODO
        pass


    def plot_residuals():
        pass
        # TODO



class DiscreteDistribution(BaseDistribution):
    """Abstract categorical model class (used as implementation base)

    TODO: More info...

    """

    def predict(self, x=None, data=None):
        """Predict discrete dependent variable for independent var samples in x.

        TODO: explain how predictions are generated using the MODE of each
        variational distribution

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predict` on a |Model|, you must first
            :meth:`.fit` it to some data.

        Parameters
        ----------
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x``.  If ``data`` is |None|, 
            it is assumed that ``x`` is a |ndarray|.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` is a string
            or list of strings containing the columns from ``data`` to use.

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

        # Process data
        x = process_data(self, x, data)

        # Predict using the mode of the mean model
        return self._session.run(
            self.mean_obj.mode(), 
            feed_dict={self._ph['x']: x,
                       self._ph['batch_size']: [x.shape[0]]})


    def calibration_curve(self, x=None, y=None, data=None,
                          split_by=None, bins=10):
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
        x : |ndarray| or int or str or list of str or int
            Independent variable values of the dataset to fit (aka the 
            "features").  If ``data`` was passed as a |DataFrame|, ``x`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        y : |ndarray| or int or str or list of str or int
            Dependent variable values of the dataset to fit (aka the 
            "reponse"). If ``data`` was passed as a |DataFrame|, ``y`` can be
            an int or string or list of ints or strings specifying the columns
            of that |DataFrame| to use as independent variables. If |None|, 
            will use the data the model was trained on (the default).
        data : |None| or |DataFrame|
            DataFrame containing ``x`` and ``y``.  If ``data`` is |None|, 
            it is assumed that ``x`` and ``y`` are |ndarray|s.  If ``data`` 
            is a |DataFrame|, it is assumed that ``x`` and ``y`` are strings
            or lists of strings containing the columns from ``data`` to use.
        split_by : int or str
            Draw curve independently for datapoints with each unique value in 
            this categorical column number.
        bins : int or list of float or |ndarray|
            Number of bins used to compute the curve.  An integer to
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

