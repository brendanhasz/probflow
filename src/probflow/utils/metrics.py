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
    "f1_score",
    "get_metric_fn",
    "mean_absolute_error",
    "mean_squared_error",
    "precision",
    "r_squared",
    "sum_squared_error",
    "true_negative_rate",
    "true_positive_rate",
]


from collections.abc import Callable

import numpy as np

from probflow.utils.casting import to_numpy
from probflow.utils.typing import TensorLike


def as_numpy(
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> Callable[[TensorLike, TensorLike], float]:
    """Cast inputs to numpy arrays and same shape before computing metric."""

    def metric_fn(y_true: TensorLike, y_pred: TensorLike) -> float:

        # Cast to numpy arrays
        y_true_np = to_numpy(y_true)
        y_pred_np = to_numpy(y_pred)

        # Ensure correct sizes
        if y_true_np.ndim == 1:
            y_true_np = np.expand_dims(y_true_np, 1)
        if y_pred_np.ndim == 1:
            y_pred_np = np.expand_dims(y_pred_np, 1)

        # Return metric function on consistent arrays
        return fn(y_true_np, y_pred_np)

    return metric_fn


@as_numpy
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy of predictions."""
    return float(np.mean(y_pred == y_true))


@as_numpy
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean(np.square(y_true - y_pred)))


@as_numpy
def sum_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sum of squared error."""
    return float(np.sum(np.square(y_true - y_pred)))


@as_numpy
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


@as_numpy
def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    ss_res = np.sum(np.square(y_true - y_pred))
    return float(1.0 - ss_res / ss_tot)


@as_numpy
def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True positive rate aka sensitivity aka recall."""
    p = np.sum(y_true == 1)
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    return float(tp / p)


@as_numpy
def true_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True negative rate aka specificity aka selectivity."""
    n = np.sum(y_true == 0)
    tn = np.sum((y_pred == y_true) & (y_true == 0))
    return float(tn / n)


@as_numpy
def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision."""
    ap = np.sum(y_pred)
    tp = np.sum((y_pred == y_true) & (y_true == 1))
    return float(tp / ap)


@as_numpy
def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F-measure."""
    p = precision(y_true, y_pred)
    r = true_positive_rate(y_true, y_pred)
    return float(2 * (p * r) / (p + r))


# TODO: jaccard_similarity


# TODO: roc_auc


# TODO: cross_entropy


def get_metric_fn(
    metric: str | Callable[[TensorLike, TensorLike], float],
) -> Callable[[TensorLike, TensorLike], float]:
    """Get a function corresponding to a metric string."""
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
