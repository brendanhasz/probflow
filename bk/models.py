"""Pre-built models.
"""

from abc import ABC, abstractmethod

class _BaseModel(ABC):
    """Abstract model class (just used as implementation base)

    TODO: More info...

    """


    def __init__(self, *args, **kwargs):
        """Construct model.

        TODO: docs
        """
        self.args = args
        self.kwargs = kwargs
        self.built_model = None


    @abstractmethod
    def build(self, data):
        """Build model.
        
        Inheriting class must define this method by building the 
        model for that class.  Should set `self.built_model` to a
        TensorFlow Probability distribution, using the inputs to 
        the constructor which are stored in `self.args` and 
        `self.kwargs`.
        """
        pass


    def fit(self, x, y, batch_size=128, epochs=100, 
            optimizer='adam', learning_rate=None, metrics=None):
        """Fit model.

        TODO: Docs...

        """
        # TODO: build the model
        # TODO: set up the data for fitting
        # TODO: fit the model


    def predict(self, x):
        """Predict dependent variable for samples in x.

        TODO: Docs...

        """
        #TODO


    def residuals(self, x, y):
        """Compute residuals between true and predicted values. 

        TODO: Docs...

        """
        #TODO


    def plot_residuals(self, x, y):
        """Plot the residual distribution.

        TODO: Docs...

        """
        #TODO


    def metrics(self, x, y, metric_list):
        """Compute prediction metrics.

        # Arguments
            x: validation independent variable samples.
            y: validation dependent variable samples.
            metric_list: list of metrics to evaluate.

        TODO: Docs...

        """
        #TODO

        
    def posterior(self, params=None):
        """Predict dependent variable for samples in x.

        TODO: Docs...

        """
        #TODO

        
    def predictive_distribution(self, x):
        """Predict dependent variable for samples in x.

        TODO: Docs...

        """
        #TODO

        
    def predictive_prc(self, x, y):
        """Predict dependent variable for samples in x.

        TODO: Docs...

        """
        #TODO

        
    def pred_dist_covered(self, x, y, prc):
        """Predict dependent variable for samples in x.

        TODO: Docs...

        """
        #TODO

        
    def pred_dist_coverage(self, x, y, prc):
        """Predict dependent variable for samples in x.

        TODO: Docs...

        """
        #TODO

        
    def calibration_curve(self, x, y):
        """Predict dependent variable for samples in x.

        TODO: Docs...

        """
        #TODO

        
    def coverage_by(self, x, y, x_by, prc):
        """Predict dependent variable for samples in x.

        TODO: Docs...

        """
        #TODO


class LinearRegression(_BaseModel):
    #TODO


class LogisticRegression(_BaseModel):
    #TODO


class DenseRegression(_BaseModel):
    #TODO


class DenseClassifier(_BaseModel):
    #TODO


class Conv1dRegression(_BaseModel):
    #TODO


class Conv1dClassifier(_BaseModel):
    #TODO


class Conv2dRegression(_BaseModel):
    #TODO


class Conv2dClassifier(_BaseModel):
    #TODO


class DenseAutoencoder(_BaseModel):
    #TODO


class Conv1dAutoencoder(_BaseModel):
    #TODO


class Conv2dAutoencoder(_BaseModel):
    #TODO

#TODO: Neural matrix factorization

#TODO: BayesianCorrelation