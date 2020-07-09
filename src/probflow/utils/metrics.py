"""Metrics.

Evaluation metrics

* :func:`.log_prob`
* :func:`.acc`
* :func:`.accuracy`
* :func:`.mse`
* :func:`.sse`
* :func:`.mae`

----------

"""


__all__ = [
    "accuracy",
    "mean_squared_error",
    "sum_squared_error",
    "mean_absolute_error",
    "r_squared",
    "true_positive_rate",
    "true_negative_rate",
    "precision",
    "f1_score",
    "get_metric_fn",
]


import numpy as np
import pandas as pd


def as_numpy(fn):
    """Cast inputs to numpy arrays and same shape before computing metric"""

    def metric_fn(y_true, y_pred):

        # Cast y_true to numpy array
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            new_y_true = y_true.values
        elif isinstance(y_true, np.ndarray):
            new_y_true = y_true
        else:
            new_y_true = y_true.numpy()

        # Cast y_pred to numpy array
        if isinstance(y_pred, (pd.Series, pd.DataFrame)):
            new_y_pred = y_pred.values
        elif isinstance(y_pred, np.ndarray):
            new_y_pred = y_pred
        else:
            new_y_pred = y_pred.numpy()

        # Ensure correct sizes
        if new_y_true.ndim == 1:
            new_y_true = np.expand_dims(new_y_true, 1)
        if new_y_pred.ndim == 1:
            new_y_pred = np.expand_dims(new_y_pred, 1)

        # Return metric function on consistent arrays
        return fn(new_y_true, new_y_pred)

    return metric_fn


@as_numpy
def accuracy(y_true, y_pred):
    """Accuracy of predictions."""
    return np.mean(y_pred == y_true)


@as_numpy
def mean_squared_error(y_true, y_pred):
    """Mean squared error."""
    return np.mean(np.square(y_true - y_pred))


@as_numpy
def sum_squared_error(y_true, y_pred):
    """Sum of squared error."""
    return np.sum(np.square(y_true - y_pred))


@as_numpy
def mean_absolute_error(y_true, y_pred):
    """Mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


@as_numpy
def r_squared(y_true, y_pred):
    """Coefficient of determination."""
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    ss_res = np.sum(np.square(y_true - y_pred))
    return 1.0 - ss_res / ss_tot


@as_numpy
def true_positive_rate(y_true, y_pred):
    """True positive rate aka sensitivity aka recall."""
    p = np.sum(y_true == 1)
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    return tp / p


@as_numpy
def true_negative_rate(y_true, y_pred):
    """True negative rate aka specificity aka selectivity."""
    n = np.sum(y_true == 0)
    tn = np.sum((y_pred == y_true) & (y_true == 0))
    return tn / n


@as_numpy
def precision(y_true, y_pred):
    """Precision."""
    ap = np.sum(y_pred)
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    return tp / ap


@as_numpy
def f1_score(y_true, y_pred):
    """F-measure."""
    p = precision(y_true, y_pred)
    r = true_positive_rate(y_true, y_pred)
    return 2 * (p * r) / (p + r)


# TODO: jaccard_similarity


# TODO: roc_auc


# TODO: cross_entropy


def get_metric_fn(metric):
    """Get a function corresponding to a metric string"""

    # List of valid metric strings
    metrics = {
        "accuracy": accuracy,
        "acc": accuracy,
        "mean_squared_error": mean_squared_error,
        "mse": mean_squared_error,
        "sum_squared_error": sum_squared_error,
        "sse": sum_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "mae": mean_absolute_error,
        "r_squared": r_squared,
        "r2": r_squared,
        "recall": true_positive_rate,
        "sensitivity": true_positive_rate,
        "true_positive_rate": true_positive_rate,
        "tpr": true_positive_rate,
        "specificity": true_negative_rate,
        "selectivity": true_negative_rate,
        "true_negative_rate": true_negative_rate,
        "tnr": true_negative_rate,
        "precision": precision,
        "f1_score": f1_score,
        "f1": f1_score,
        # 'jaccard_similarity': jaccard_similarity,
        # 'jaccard': jaccard_similarity,
        # 'roc_auc': roc_auc,
        # 'auroc': roc_auc,
        # 'auc': roc_auc,
    }

    # Return the corresponding function
    if callable(metric):
        return metric
    elif isinstance(metric, str):
        if metric not in metrics:
            raise ValueError(
                metric
                + " is not a valid metric string. "
                + "Valid strings are: "
                + ", ".join(metrics.keys())
            )
        else:
            return metrics[metric]
    else:
        raise TypeError("metric must be a str or callable")
