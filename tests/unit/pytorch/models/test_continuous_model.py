import matplotlib.pyplot as plt
import numpy as np
import pytest

from probflow.data import make_generator
from probflow.distributions import Normal
from probflow.models import ContinuousModel
from probflow.parameters import Parameter, ScaleParameter
from probflow.utils.casting import to_tensor


def test_ContinuousModel(plot):
    """Tests probflow.models.ContinuousModel"""

    class MyModel(ContinuousModel):
        def __init__(self):
            self.weight = Parameter([5, 1], name="Weight")
            self.bias = Parameter([1, 1], name="Bias")
            self.std = ScaleParameter([1, 1], name="Std")

        def __call__(self, x):
            x = to_tensor(x)
            return Normal(x @ self.weight() + self.bias(), self.std())

    # Instantiate the model
    model = MyModel()

    # Data
    x = np.random.randn(100, 5).astype("float32")
    w = np.random.randn(5, 1).astype("float32")
    y = x @ w + 1

    # Fit the model
    model.fit(x, y, batch_size=50, epochs=100, lr=0.01)

    # predictive intervals
    lb, ub = model.predictive_interval(x[:22, :])
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 22
    assert lb.shape[1] == 1
    assert ub.ndim == 2
    assert ub.shape[0] == 22
    assert ub.shape[1] == 1

    # predictive intervals lower ci
    llb = model.predictive_interval(x[:22, :], side="lower")
    assert isinstance(llb, np.ndarray)
    assert llb.ndim == 2
    assert llb.shape[0] == 22
    assert llb.shape[1] == 1
    assert np.all(llb <= ub)

    # predictive intervals upper ci
    uub = model.predictive_interval(x[:22, :], side="upper")
    assert isinstance(uub, np.ndarray)
    assert uub.ndim == 2
    assert uub.shape[0] == 22
    assert uub.shape[1] == 1
    assert np.all(uub >= lb)
    assert np.all(uub >= llb)

    # predictive intervals with batching
    lb, ub = model.predictive_interval(x[:21, :], batch_size=7)
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 21
    assert lb.shape[1] == 1
    assert ub.ndim == 2
    assert ub.shape[0] == 21
    assert ub.shape[1] == 1

    # aleatoric intervals
    lb, ub = model.aleatoric_interval(x[:23, :])
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 23
    assert lb.shape[1] == 1
    assert ub.ndim == 2
    assert ub.shape[0] == 23
    assert ub.shape[1] == 1

    # epistemic intervals
    lb, ub = model.epistemic_interval(x[:24, :])
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.ndim == 2
    assert lb.shape[0] == 24
    assert lb.shape[1] == 1
    assert ub.ndim == 2
    assert ub.shape[0] == 24
    assert ub.shape[1] == 1

    # posterior predictive plot with one sample
    model.pred_dist_plot(x[:1, :])
    if plot:
        plt.title("Should be one dist on one subfig")
        plt.show()

    # posterior predictive plot with one sample, showing ci
    model.pred_dist_plot(x[:1, :], ci=0.95, style="hist")
    if plot:
        plt.title("Should be one dist on one subfig, w/ ci=0.95")
        plt.show()

    # posterior predictive plot with two samples
    model.pred_dist_plot(x[:2, :])
    if plot:
        plt.title("Should be two dists on one subfig")
        plt.show()

    # posterior predictive plot with two samples, two subfigs
    model.pred_dist_plot(x[:2, :], individually=True)
    if plot:
        plt.title("Should be two dists on two subfigs")
        plt.show()

    # posterior predictive plot with six samples, 6 subfigs, 2 cols
    model.pred_dist_plot(x[:6, :], individually=True, cols=2)
    if plot:
        plt.title("Should be 6 dists, 6 subfigs, 2 cols")
        plt.show()

    # predictive prc
    prcs = model.predictive_prc(x[:7, :], y[:7, :])
    assert isinstance(prcs, np.ndarray)
    assert prcs.ndim == 2
    assert prcs.shape[0] == 7
    assert prcs.shape[1] == 1

    with pytest.raises(TypeError):
        prcs = model.predictive_prc(x[:7, :], None)

    # predictive distribution covered for each sample
    cov = model.pred_dist_covered(x[:11, :], y[:11, :])
    assert isinstance(cov, np.ndarray)
    assert cov.ndim == 2
    assert cov.shape[0] == 11
    assert cov.shape[1] == 1

    with pytest.raises(ValueError):
        cov = model.pred_dist_covered(x, y, n=-1)
    with pytest.raises(ValueError):
        cov = model.pred_dist_covered(x, y, ci=-0.1)
    with pytest.raises(ValueError):
        cov = model.pred_dist_covered(x, y, ci=1.1)

    # predictive distribution covered for each sample
    cov = model.pred_dist_coverage(x[:11, :], y[:11, :])
    assert isinstance(cov, np.float)

    # plot coverage by
    xo, co = model.coverage_by(x[:, :1], x, y)
    assert isinstance(xo, np.ndarray)
    assert isinstance(co, np.ndarray)
    if plot:
        plt.title("should be coverage by plot")
        plt.show()

    # r squared
    r2 = model.r_squared(x, y, n=21)
    assert isinstance(r2, np.ndarray)
    assert r2.ndim == 2
    assert r2.shape[0] == 21
    assert r2.shape[1] == 1

    # r squared with an ArrayDataGenerator
    dg = make_generator(x, y)
    r2 = model.r_squared(dg, n=22)
    assert isinstance(r2, np.ndarray)
    assert r2.ndim == 2
    assert r2.shape[0] == 22
    assert r2.shape[1] == 1

    # plot the r2 dist
    model.r_squared_plot(x, y, style="hist")
    if plot:
        plt.title("should be r2 dist")
        plt.show()

    # residuals
    res = model.residuals(x, y)
    assert isinstance(res, np.ndarray)
    assert res.ndim == 2
    assert res.shape[0] == 100
    assert res.shape[1] == 1

    # plot the distribution of residuals
    model.residuals_plot(x, y)
    if plot:
        plt.title("should be residuals dist")
        plt.show()

    # calibration curve
    p, p_hat = model.calibration_curve(x[:90, :], y[:90, :], resolution=11)
    assert isinstance(p, np.ndarray)
    assert isinstance(p_hat, np.ndarray)
    assert p.ndim == 1
    assert p.shape[0] == 11
    assert p_hat.ndim == 1
    assert p_hat.shape[0] == 11
    assert np.all(p >= 0)
    assert np.all(p <= 1)
    assert np.all(p_hat >= 0)
    assert np.all(p_hat <= 1)

    # calibration curve (with batching)
    p, p_hat = model.calibration_curve(
        x[:90, :], y[:90, :], resolution=11, batch_size=30
    )
    assert isinstance(p, np.ndarray)
    assert isinstance(p_hat, np.ndarray)
    assert p.ndim == 1
    assert p.shape[0] == 11
    assert p_hat.ndim == 1
    assert p_hat.shape[0] == 11
    assert np.all(p >= 0)
    assert np.all(p <= 1)
    assert np.all(p_hat >= 0)
    assert np.all(p_hat <= 1)

    # calibration curve plot
    model.calibration_curve_plot(x, y, resolution=11)
    if plot:
        plt.title("should be calibration curve")
        plt.show()

    # calibration curve plot (with batching)
    model.calibration_curve_plot(x, y, resolution=11, batch_size=25)
    if plot:
        plt.title("should be calibration curve (with batching)")
        plt.show()

    # calibration metrics: msce
    msce = model.calibration_metric(
        "msce", x[:90, :], y[:90, :], resolution=11
    )
    assert isinstance(msce, float)
    assert msce >= 0
    assert msce <= 1

    # calibration metrics: rmsce
    rmsce = model.calibration_metric(
        "rmsce", x[:90, :], y[:90, :], resolution=11
    )
    assert isinstance(rmsce, float)
    assert rmsce >= 0
    assert rmsce <= 1

    # calibration metrics: mace
    mace = model.calibration_metric(
        "mace", x[:90, :], y[:90, :], resolution=11
    )
    assert isinstance(mace, float)
    assert mace >= 0
    assert mace <= 1

    # calibration metrics: ma
    ma = model.calibration_metric("ma", x[:90, :], y[:90, :], resolution=11)
    assert isinstance(ma, float)
    assert ma >= 0
    assert ma <= 1

    # should raise value error on invalid metric name
    with pytest.raises(ValueError):
        ma = model.calibration_metric(
            "lala", x[:90, :], y[:90, :], resolution=11
        )

    # calibration metrics: list of em
    metrics = model.calibration_metric(
        ["mace", "ma"], x[:90, :], y[:90, :], resolution=11
    )
    assert isinstance(metrics, dict)
    assert len(metrics) == 2
    assert "mace" in metrics
    assert "ma" in metrics
    assert metrics["mace"] >= 0
    assert metrics["mace"] <= 1
    assert metrics["ma"] >= 0
    assert metrics["ma"] <= 1

    # calibration metric with batching
    msce = model.calibration_metric(
        "msce", x[:90, :], y[:90, :], resolution=11, batch_size=30
    )
    assert isinstance(msce, float)
    assert msce >= 0
    assert msce <= 1

    # sharpness
    sha = model.sharpness(x[:90, :])
    assert isinstance(sha, (float, np.floating))
    assert rmsce >= 0

    # sharpness w/ batching
    sha = model.sharpness(x[:90, :], batch_size=30)
    assert isinstance(sha, (float, np.floating))
    assert rmsce >= 0

    # dispersion metric: cv
    dm = model.dispersion_metric("cv", x[:90, :])
    assert isinstance(dm, (float, np.floating))
    assert dm >= 0

    # dispersion metric: qcd
    dm = model.dispersion_metric("qcd", x[:90, :])
    assert isinstance(dm, (float, np.floating))
    assert dm >= 0

    # dispersion metric w/ batching
    dm = model.dispersion_metric("cv", x[:90, :], batch_size=30)
    assert isinstance(dm, (float, np.floating))
    assert dm >= 0

    # should raise value error on invalid metric name
    with pytest.raises(ValueError):
        dm = model.dispersion_metric("lala", x[:90, :])

    # dispersion metrics: list of em
    metrics = model.dispersion_metric(["cv", "qcd"], x[:90, :])
    assert isinstance(metrics, dict)
    assert len(metrics) == 2
    assert "cv" in metrics
    assert "qcd" in metrics
    assert metrics["cv"] >= 0
    assert metrics["qcd"] >= 0
