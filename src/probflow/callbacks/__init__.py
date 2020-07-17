"""
The callbacks module contains classes for monitoring and adjusting the
training process.

* :class:`.Callback` - abstract base class for all callbacks
* :class:`.EarlyStopping` - stop training if some metric stops improving
* :class:`.KLWeightScheduler` - set the KL weight by epoch
* :class:`.LearningRateScheduler` - set the learning rate by epoch
* :class:`.MonitorELBO` - record the ELBO loss over the course of training
* :class:`.MonitorMetric` - record a metric over the course of training
* :class:`.MonitorParameter` - record a parameter over the course of training
* :class:`.TimeOut` - stop training after a certain amount of time

----------

"""


__all__ = [
    "Callback",
    "EarlyStopping",
    "KLWeightScheduler",
    "LearningRateScheduler",
    "MonitorELBO",
    "MonitorMetric",
    "MonitorParameter",
    "TimeOut",
]


from .callback import Callback
from .early_stopping import EarlyStopping
from .kl_weight_scheduler import KLWeightScheduler
from .learning_rate_scheduler import LearningRateScheduler
from .monitor_elbo import MonitorELBO
from .monitor_metric import MonitorMetric
from .monitor_parameter import MonitorParameter
from .time_out import TimeOut
