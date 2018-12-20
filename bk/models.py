"""Pre-built models.
"""

#imports...

class _BaseModel():
    """Abstract model class (just used as implementation base)


    More info...


    """

    def __init__(self, ):
        """Construct model.

        Args:
          ...
        """
        self.built_model = None

    def build(self, data):
        """Build model.
        
        ...
        """
        # TODO: build child models
        # TODO: Construct this from built child models

    def fit(self, x, y, batch_size=128, epochs=100, 
            optimizer='adam', learning_rate=None, metrics=None):
        """Fit model.

        Docs...
        """
        # TODO: build the model
        # TODO: set up the data for fitting
        # TODO: fit the model

    # TODO:
    """
    - predict(x)
    - residuals(x, y)
    - plot_residuals(x, y)
    - metrics(x, y, metric_list)
    - posterior(params=None)
    - predictive_distribution(x)
    - predictive_prc(x, y)
    - pred_dist_covered(x, y, prc)
    - pred_dist_coverage(x, y, prc)
    - calibration_curve(x, y)
    - coverage_by(x, y, x_by, prc)
    """


class LinearRegression(_BaseModel):