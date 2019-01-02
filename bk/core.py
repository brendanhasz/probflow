"""Abstract classes.

TODO: more info...

"""



from abc import ABC, abstractmethod
from scipy import stats
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class BaseVariable(ABC):
    """Abstract Variable class (used as an implementation base)"""
    pass



class BaseLayer(ABC):
    """Abstract layer class (used as an implementation base)

    This is an abstract base class for a layer.  Layers are objects which take 
    other objects as input (can be either other layers or tensors) and output a
    tensor.  A layer's inputs 


    Required attributes and methods
    -------------------------------
    An inheriting class must define the following properties and methods:
    * `default_args` (attribute)
    * `_build` (method)
    * `_log_loss` (method)

    The `default_args` attribute should contain a dict whose keys are the names
    of the layer's arguments, and whose values are the default value of each 
    argument.  Setting an argument's value in `default_args` to `None` causes
    that argument to be mandatory (TypeError if that argument's value is not 
    specified when instantiating the class).

    The `_build` method should return a `Tensor` which was built from this 
    layer's arguments. TODO: more details...

    The `_log_loss` method should return the log loss incurred by this layer. TODO: more...


    See Also
    --------
    func : why you should see it


    Notes
    -----
    TODO: Docs...


    Examples
    --------
    We can define a layer which adds two arguments like so:
    ```
    class Add(BaseLayer):

        self.default_args = {
            'a': None,
            'b': None
        }

        def _build(self, args, data):
            return args['a'] + args['b']

        def _log_loss(self, obj, vals):
            return 0
    ```

    then we can use that layer to add two other layers or tensors:

    ```
    x = Input()
    b = Variable()
    mu = Add(x, b)
    ```

    For more examples, see Add, Sub, Mul, Div, Abs, Exp, and Log in layers.py

    """


    @property
    @abstractmethod
    def default_args(self):
        """Layer parameters and their default values.

        Inheriting class must define this attribute as a dict whose keys are 
        the names of the layer's arguments, and whose values are the default 
        value of each argument.  Setting an argument's value in `default_args` 
        to `None` causes that argument to be mandatory (TypeError if that 
        argument's value is not specified when instantiating the class).
        """
        pass


    def __init__(self, *args, **kwargs):
        """Construct layer.

        TODO: docs. Mention that actually building the tf graph is
        delayed until build() or fit() is called.
        """

        # Set layer arguments, using args, kwargs, and defaults 
        for ix, arg in enumerate(self.default_args):
            if ix<len(args):
                self.args[arg] = args[ix]
            elif arg in kwargs:
                self.args[arg] = kwargs[arg]
            else:
                self.args[arg] = self.default_args[arg]

        # Ensure all required arguments have been set
        if None in self.args.values():
            raise TypeError('required arg(s) were not set. '+
                            type(self).__name__+' requires args: '+
                            ', '.join(self.default_args.keys())) 

        # Ensure all arguments are of correct type
        for arg in self.args:
            if not self._arg_is('valid', arg):
                msg = ('Invalid type for ' + type(self).__name__ + 
                       'argument ' + arg + '. Must be one of: int, float, ' + 
                       'np.ndarray, tf.Tensor, or a bk layer, model, ' +
                       'or distribution.')
                raise TypeError(msg)

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


    @abstractmethod
    def _log_loss(self, obj, vals):
        """Compute the log loss incurred by this layer.
        
        Inheriting class must define this method by computing the loss of this
        layer (and only this layer, not its args!).

        TODO: docs...

        """
        pass


    def _arg_is(self, type_str, arg):
        """Return true if arg is of type type_str."""
        if type_str=='number':
            return isinstance(arg, (int, float, np.ndarray))
        elif type_str=='tensor':
            return isinstance(arg, tf.Tensor)
        elif type_str=='tensor_like':
            return isinstance(arg, (int,float,np.ndarray,tf.Tensor))
        elif type_str=='model':
            return isinstance(arg, BaseModel)
        elif type_str=='layer':
            return isinstance(arg, BaseLayer)
        elif type_str=='variable':
            return isinstance(arg, BaseVariable)
        elif type_str=='valid':
            return isinstance(arg, (int, float, np.ndarray,
                                    tf.Tensor, BaseModel, BaseLayer))
        else:
            raise TypeError('type_str must a string, one of: number, tensor,' +
                            ' tensor_like, model, layer, or valid')


    def build_args(self, data):
        """Build each of this layer's arguments."""
        for arg, arg_name in self.args.items():
            if _arg_is('tensor_like', arg):
                self.built_args[arg_name] = arg
                self.mean_args[arg_name] = arg
            elif _arg_is('model', arg):
                arg.build(data)
                self.built_args[arg_name] = arg.built_obj.sample()
                self.mean_args[arg_name] = arg.mean_obj.mean()
            elif _arg_is('layer', arg):
                arg.build(data)
                self.built_args[arg_name] = arg.built_obj
                self.mean_args[arg_name] = arg.mean_obj
            elif _arg_is('variable', arg):
                arg.build(data)
                self.built_args[arg_name] = arg.sample()
                self.mean_args[arg_name] = arg.mean()


    def sum_arg_losses(self):
        """Sum the loss of all this layer's arguments."""
        self.arg_loss_sum = 0
        self.mean_arg_loss_sum = 0
        for arg, arg_name in self.args.items():
            if _arg_is('tensor_like', arg):
                pass #no loss incurred by tensors
            elif _arg_is('layer', arg):
                self.arg_loss_sum += (
                    arg.arg_loss_sum + 
                    arg._log_loss(arg.built_obj, self.built_args[arg_name]))
                self.mean_arg_loss_sum += (
                    arg.mean_arg_loss_sum + 
                    arg._log_loss(arg.mean_obj, self.mean_args[arg_name]))
            elif _arg_is('variable', arg):
                self.arg_loss_sum += arg.log_loss(self.built_args[arg_name])
                self.mean_arg_loss_sum += arg.log_loss(self.mean_args[arg_name])


    def build(self, data):
        """Build this layer's arguments and loss, and then build the layer.

        TODO: actually do docs for this one...

        """
        self.build_args(data)
        self.sum_arg_losses()
        self.built_obj = self._build(self.built_args, self.data)
        self.mean_obj = self._build(self.mean_args, self.data)


    # TODO: fix the circular imports so these can work:
    #def __add__(self, other):
    #    """Add this layer to another layer, variable, or value."""
    #    return Add(self, other)


    #def __sub__(self, other):
    #    """Subtract from this layer another layer, variable, or value."""
    #    return Sub(self, other)


    #def __mul__(self, other):
    #   """Multiply this layer by another layer, variable, or value."""
    #    return Mul(self, other)


    #def __div__(self, other):
    #    """Divide this layer by another layer, variable, or value."""
    #    return Div(self, other)


    #def __abs__(self, other):
    #    """Take the absolute value of the input to this layer."""
    #    return Abs(self)



class BaseModel(BaseLayer):
    """Abstract model class (just used as an implementation base)

    TODO: More info...
    talk about how a model defines a parameterized probability distribution which
    you can call fit on

    """


    def fit(self, x, y, batch_size=128, epochs=100, 
            optimizer='adam', learning_rate=None, metrics=None):
        """Fit model.

        TODO: Docs...

        """

        # Set up the data for fitting
        # TODO
        #y_vals = iterator...
        #x_vals = iterator...

        # Recursively build this model and its args
        self.build(data)
        model = self.built_obj

        # Set up TensorFlow graph for the losses
        self.log_loss = (self.arg_loss_sum + 
                         self._log_loss(self.built_obj, y_vals))
        self.mean_log_loss = (self.mean_arg_loss_sum + 
                              self._log_loss(self.mean_obj, y_vals))

        # TODO: fit the model

        self.is_fit = True
        pass


    def ensure_is_fit(self):
        """Raises a RuntimeError if model has not yet been fit."""
        if not self.is_fit:
            raise RuntimeError('model must first be fit') 


    def posterior(self, params=None):
        """Draw samples from parameter posteriors.

        TODO: Docs...

        """

        # Check model has been fit
        self.ensure_is_fit()

        #TODO
        pass

        
    def predictive_distribution(self, x, num_samples=1000):
        """Draw samples from the model given x.

        TODO: Docs...

        Returns array of shape (x.shape[0],y.shape[1],num_samples)

        """

        # Check model has been fit
        self.ensure_is_fit()

        #TODO: actually compute the samples
        pass


    def predict(self, x):
        """Predict dependent variable for samples in x.

        TODO: explain how predictions are generated using the mean of each
        variational distribution

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
            a prediction when ``method=='prc'`. Between 0 and 100
            inclusive (default=50).
        num_samples : int
            Number of samples to draw from the posterior predictive
            distribution (default=100).

        Returns
        -------
        predictions : np.ndarray
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
        self.ensure_is_fit()

        # Compute predictive distribution
        pred_dist = self.predictive_distribution(x, num_samples)

        #TODO: should really do straight in tf, 
        # not tf->numpy->sample

        # TODO: and should use mean model, not sampling

        # Compute prediction based on the predictive distribution
        if method=='mean':
            return np.mean(pred_dist, axis=2)
        elif method=='median':
            return np.median(pred_dist, axis=2)
        elif method=='mode':
            return stats.mode(pred_dist, axis=2)
        elif method=='min':
            return np.amin(pred_dist, axis=2)
        elif method=='max':
            return np.amax(pred_dist, axis=2)
        elif method=='prc':
            return np.percentile(pred_dist, prc, axis=2)
        else:
            raise ValueError('method '+str(method)+' is invalid')


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


        TODO: docs...

        if individually is True, returns prob for each sample individually
            so return shape is (x.shape[0],?)
        if individually is False, returns product of each individual prob
            so return shape is (1,?)
        if dist is True, returns log probability posterior distribution
            (distribution of probs for lots of samples from the model) 
            so return shape is (?,num_samples)
        if dist is False, returns log posterior prob assuming each variable 
            takes the mean value of its variational distribution
            so return shape iss (?,1)

        """

        # Check model has been fit
        self.ensure_is_fit()

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

        also, this should probably use log_prob, above, then exp it...
        """

        # TODO: evaluate log_prob w/ tf like in log_prob above
        pass


    def prob_by(self, x, y, x_by, bins=100, plot=True):
        """Plot the probability of observations `y` given `x` and the model 
        as a function of independent variable(s) `x_by`.

        TODO: docs...
        
        """
        
        # TODO: same idea as log_prob_by above
        pass


    def cdf(self, x, y):
        """Compute the cumulative probability of `y` given `x` and the model.

        TODO: docs...

        """

        # TODO: same idea as log_prob above
        pass


    def cdf_by(self, x, y, x_by, bins=100):
        """Plot the cumulative probability of observations `y` given `x` and 
        the model as a function of independent variable(s) `x_by`.

        TODO: docs...
        
        """

        # TODO: same idea as log_prob_by above
        pass


    def log_cdf(self, x, y):
        """Compute the log cumulative probability of `y` given `x` and the model.

        TODO: docs...

        """

        # TODO: same idea as log_prob above
        pass


    def log_cdf_by(self, x, y, x_by, bins=100):
        """Plot the log cumulative probability of observations `y` given `x` 
        and the model as a function of independent variable(s) `x_by`.

        TODO: docs...
        
        """

        # TODO: same idea as log_prob_by above
        pass



class ContinuousModel(BaseModel):
    """Abstract continuous model class (used as implementation base)

    TODO: More info...

    """


    def predictive_prc(self, x, y):
        """Compute the percentile of each observation along the posterior 
        predictive distribution.

        TODO: Docs...

        """

        #TODO
        pass


    def confidence_intervals(self, x, prcs=[2.5, 97.5], num_samples=1000):
        """Compute confidence intervals on predictions for `x`.

        TODO: docs, prcs contains percentiles of predictive_distribution to use

        Parameters
        ----------
        x : np.ndarray
            Input array of independent variable values.  Should be of shape 
            (N,D), where N is the number of samples and D is the number of 
            dimensions of the independent variable.
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
        self.ensure_is_fit()

        # Compute percentiles of the predictive distribution
        pred_dist = self.predictive_distribution(x, num_samples=num_samples)
        return np.percentile(pred_dist, prcs)

        
    def pred_dist_covered(self, x, y, prc):
        """Compute whether each observation was covered by the 
        inner `prc` percentile of the posterior predictive 
        distribution.

        TODO: Docs...

        """

        # Check model has been fit
        self.ensure_is_fit()

        #TODO
        pass

        
    def pred_dist_coverage(self, x, y, prc):
        """Compute the coverage of the inner `prc` percentile of the
        posterior predictive distribution.

        TODO: Docs...
        returns a scalar (from 0 to 100)

        """

        # Check model has been fit
        self.ensure_is_fit()

        #TODO
        pass


    def coverage_by(self, x, y, x_by, prc, bins=100, plot=True):
        """Compute and plot the coverage of the inner `prc` 
        percentile of the posterior predictive distribution as a 
        function of specified independent variables.

        TODO: Docs...
        x_by should be int or length-2 list of ints which specifies what column of x to plot by
        returns x and coverage matrix
        """

        # Compute whether each sample was covered by the interval
        covered = self.pred_dist_covered(x, y, prc)

        # TODO: alternatively, x_by should be able to be any array_like
        # as long as it's same size as x.shape[0]
        
        # Plot probability as a fn of x_by cols of x
        px, py = self.plot_by(x[:,x_by], covered, 
                               bins=bins, plot=plot)

        return px, py


    def calibration_curve(self, x, y, split_by=None, bins=10):
        """Plot and return calibration curve.

        Plots and returns the calibration curve (the percentile of 
        the posterior predictive distribution on the x-axis, and the
        percent of samples which actually fall into that range on
        the y-axis).

        Parameters
        ----------
        x : np.ndarray
            Input array of independent variable values.  Should be
            of shape (N,D), where N is the number of samples and D
            is the number of dimensions of the independent variable.
        y : np.ndarray
            Array of dependent variable values.  Should be of shape
            (N,D_out), where N is the number of samples (equal to
            `x.shape[0]) and D_out is the number of dimensions of
            the dependent variable.
        split_by : int
            Draw the calibration curve independently for datapoints
            with each unique value in `x[:,split_by]` (a categorical
            column).
        bins : int or vector_like
            Bins used to compute the curve.  If an integer, will use
            `bins` evenly-spaced bins from 0 to 1.  If a vector,
            `bins` is the vector of bin edges.          

        Returns
        -------
        cx : np.ndarray
            Vector of percentiles (the middle of each percentile
            bin).  Length is determined by `bins`.
        cy : np.ndarray
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
        self.ensure_is_fit()

        #TODO
        pass



class CategoricalModel(BaseModel):
    """Abstract categorical model class (used as implementation base)

    TODO: More info...

    """


    def calibration_curve(self, x, y, split_by=None, bins=10):
        """Plot and return calibration curve.

        Plots and returns the calibration curve (estimated 
        probability of outcome vs the true probability of that 
        outcome).

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
        self.ensure_is_fit()

        #TODO
        pass


    # TODO: are there categorical equivalents of predictive_prc, 
    # pred_dist_covered, pred_dist_coverage, and coverage_by?


# TODO: DiscreteModel (for poisson etc)