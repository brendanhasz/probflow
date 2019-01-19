"""Abstract classes.

TODO: more info...

----------

"""



from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp



# Sentinel object for required arguments
REQUIRED = object()



class BaseParameter(ABC):
    """Abstract parameter class (used as an implementation base)"""
    pass



class BaseLayer(ABC):
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
            if ix<len(args):
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
            if len(args)>(len(self._default_args)+ix): #leftover args!
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
    def _build(self, args, data):
        """Build layer.
        
        Inheriting class must define this method by building the layer for that
        class.  Should return a `Tensor` or a `tfp.distribution` using the
        layer arguments in args (a dict).

        TODO: docs...

        """
        pass


    def _build_mean(self, args, data):
        """Build the layer with mean parameters.

        TODO: docs. default is to just do the same thing as _build

        """
        return self._build(args, data)


    def _log_loss(self, obj, vals):
        """Compute the log loss incurred by this layer.

        TODO: docs... default is no loss but can override when there is

        """
        return 0


    def _mean_log_loss(self, obj, vals):
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
        if type_str=='tensor_like':
            return isinstance(arg, (int, float, np.ndarray,
                                    tf.Tensor, tf.Variable))
        elif type_str=='distribution':
            return isinstance(arg, BaseDistribution)
        elif type_str=='layer':
            return isinstance(arg, BaseLayer)
        elif type_str=='parameter':
            return isinstance(arg, BaseParameter)
        elif type_str=='valid': #valid input to a layer
            return (not isinstance(arg, BaseDistribution) and
                    isinstance(arg, (int, float, np.ndarray, 
                                     tf.Tensor, tf.Variable, 
                                     BaseLayer, BaseParameter)))
        else:
            raise TypeError('type_str must a string, one of: number, tensor,' +
                            ' tensor_like, model, layer, or valid')


    def build(self, data=None):
        """Build this layer's arguments and loss, and then build the layer.

        TODO: actually do docs for this one...

        """

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
            elif self._arg_is('layer', arg):
                arg.build(data)
                self.built_args[arg_name] = arg.built_obj
                self.mean_args[arg_name] = arg.mean_obj
            elif self._arg_is('parameter', arg):
                arg._build(data)
                self.built_args[arg_name] = arg._sample(data)
                self.mean_args[arg_name] = arg._mean(data)

        # TODO: could just make parameter and layer have same interface, ie
        # built_obj and mean_obj are created during call to .build()

        # Sum the losses of this layer's arguments
        self.arg_loss_sum = 0       # log posterior probability of sample model
        self.mean_arg_loss_sum = 0  # log posterior probability of mean model
        self.kl_loss_sum = 0 # sum of KL div between variational post and priors
        for arg, arg_name in self.args.items():
            if self._arg_is('tensor_like', arg):
                pass #no loss incurred by data
            elif self._arg_is('layer', arg):
                self.arg_loss_sum += (
                    arg.arg_loss_sum + 
                    arg._log_loss(arg.built_obj, self.built_args[arg_name]))
                self.mean_arg_loss_sum += (
                    arg.mean_arg_loss_sum + 
                    arg._mean_log_loss(arg.mean_obj, self.mean_args[arg_name]))
                self.kl_loss_sum += (
                    arg.kl_loss_sum +
                    arg._kl_loss())
            elif self._arg_is('parameter', arg):
                self.arg_loss_sum += arg._log_loss(self.built_args[arg_name])
                self.mean_arg_loss_sum += arg._log_loss(self.mean_args[arg_name])
                self.kl_loss_sum += arg._kl_loss()

        # TODO: same idea - could make parameter + layer have same interface

        # Build this layer's sample model and mean model
        self.built_obj = self._build(self.built_args, data)
        self.mean_obj = self._build_mean(self.mean_args, data)


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



class BaseDistribution(BaseLayer):
    """Abstract distribution class (used as an implementation base)

    TODO: More info...
    talk about how a model defines a parameterized probability distribution which
    you can call fit on

    """


    def _log_loss(self, obj, vals):
        """Compute the log loss ."""
        return obj.log_prob(vals)


    def fit(self, x, y, batch_size=128, epochs=100, 
            optimizer='adam', learning_rate=None, metrics=None):
        """Fit model.

        TODO: Docs...

        TODO: add math about variational inference + elbo loss func
        TODO: and make sure to reference: https://arxiv.org/pdf/1505.05424.pdf
        (just to say that the loss used is -ELBO + log_likelihood)

        x and y should be able to be numpy arrays (in which case will set up as tf.Dataset automatically)
        or pandas dfs
        but they should also be able to be passed as tf.Datasets directly
        how to get that to work with Input tho?

        metrics='mae' or 'accuracy' or something

        Fit func should also have a validation_split arg (which defines what percentage of the data to use as validation data, along w/ shuffle arg, etc)
        Fit func should automatically normalize the input data and try to transform it (but can disable that w/ auto_normalize=False, auto_transform=False)
        And convert to float32s (but can change w/ dtype=tf.float64 or something)
        Also should have callbacks like let's to do early stopping etc
        Also have a monitor param which somehow lets you monitor the value of parameters over training?
        And make sure to make it sklearn-compatible (ie you can put it in a pipeline and everything)
        should have a show_tensorboard option (default is False) https://www.tensorflow.org/guide/summaries_and_tensorboard


        """

        # Set up the data for fitting
        # TODO
        #y_vals = iterator...
        #x_vals = iterator...
        # N = number of datapoints (x.shape[0])

        # Recursively build this model and its args
        self.build(data)
        model = self.built_obj

        # Set up TensorFlow graph for the losses
        self.log_loss = (self.arg_loss_sum + 
                         self._log_loss(self.built_obj, y_vals))
        self.mean_log_loss = (self.mean_arg_loss_sum + 
                              self._log_loss(self.mean_obj, y_vals))
        self.kl_loss = (self.kl_loss_sum + 
                        self._kl_loss())  #TODO: tho a layer shouldn't really have priors?

        # Loss functions
        log_likelihood = tf.reduce_mean(model.log_prob(y_vals))
        kl_loss = self.kl_loss / N
        elbo_loss = kl_loss - log_likelihood

        # TODO: fit the model

        self.is_fit = True
        pass


    def _ensure_is_fit(self):
        """Raises a RuntimeError if model has not yet been fit."""
        if not self.is_fit:
            raise RuntimeError('model must first be fit') 


    def posterior(self, params=None):
        """Draw samples from parameter posteriors.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.posterior` on a |Model|, you must first 
            :meth:`.fit` it to some data.

        """

        # Check model has been fit
        self._ensure_is_fit()

        #TODO
        pass

        
    def predictive_distribution(self, x, num_samples=1000):
        """Draw samples from the model given x.

        TODO: Docs...

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predictive_distribution` on a |Model|, you 
            must first :meth:`.fit` it to some data.

        Returns array of shape (x.shape[0],y.shape[1],num_samples)

        """

        # Check model has been fit
        self._ensure_is_fit()

        #TODO: actually compute the samples
        pass


    def predict(self, x):
        """Predict dependent variable for samples in x.

        TODO: explain how predictions are generated using the mean of each
        variational distribution

        .. admonition:: Model must be fit first!

            Before calling :meth:`.predict` on a |Model|, you must first 
            :meth:`.fit` it to some data.

        TODO: update parameters list below which is wrong

        Parameters
        ----------
        x : np.ndarray
            Input array of independent variable values.  Should be
            of shape (N,D), where N is the number of samples and D
            is the number of dimensions of the independent variable.
        method : {'mean', 'median', 'mode', 'min', 'max', 'prc'}
            Method to use to convert the predictive distribution 
            into a single prediction.

            * 'mean': Use the mean of the predictive distribution
              (the default).
            * 'median': Use the median of the predictive dist.
            * 'mode': Use the mode of the predictive dist.
            * 'min': Use the minimum of the predictive dist.
            * 'max': Use the maximum of the predictive dist.
            * 'prc': Use a specified percentile of the predictive 
              dist.  The `prc` arg sets what percentile value to 
              use.

        prc : float
            Percentile of the predictive distribution to use as 
            a prediction when ``method=='prc'``. Between 0 and 100
            inclusive (default=50).
        num_samples : int
            Number of samples to draw from the posterior predictive
            distribution (default=100).

        Returns
        -------
        |ndarray|
            Predicted y-value for each sample in `x`.  Will be of
            size (N,D_out), where N is the number of samples (equal
            to `x.shape[0]`) and D_out is the number of output 
            dimensions (equal to `y.shape[1]`).

        See Also
        --------
        predictive_distribution : used to generate the distribution
            which is used to make the prediction.

        Notes
        -----
        TODO: Docs...

        Examples
        --------
        TODO: Docs...

        """

        # Check model has been fit
        self._ensure_is_fit()

        # TODO: run tf session + predict using mean model



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
        px, py = self.plot_by(x[:,x_by], probs, 
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

    def metrics(self, x=None, y=None, metric_list=[]):
        """Compute metrics of model performance on data

        TODO: docs, metric_list can contain 'mse', 'mae', 'accuracy', etc

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

        """



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
        px, py = self.plot_by(x[:,x_by], covered, 
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

