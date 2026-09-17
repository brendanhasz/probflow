"""Helper functions for backend-specific tasks during training."""

from typing import TYPE_CHECKING, Any

from probflow.utils.settings import ProbflowBackend, Sampling, get_backend
from probflow.utils.typing import BackendVariable

if TYPE_CHECKING:
    from probflow.models.model import Model


class JaxAdam:
    """A minimal Adam optimizer operating on a list of JaxVariables."""

    def __init__(
        self,
        trainable_variables: list[Any],
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
    ) -> None:
        import jax.numpy as jnp

        self.variables = list(trainable_variables)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._t = 0
        self._m = [jnp.zeros_like(v.value) for v in self.variables]
        self._v = [jnp.zeros_like(v.value) for v in self.variables]

    def zero_grad(self) -> None:
        """No-op: JAX has no persistent gradient state to clear."""

    def step(self, grads: list[Any]) -> None:
        """Update each variable in place given a matching list of gradients."""
        import jax.numpy as jnp

        self._t += 1
        bias_correction_1 = 1 - self.beta_1**self._t
        bias_correction_2 = 1 - self.beta_2**self._t
        for i, (var, grad) in enumerate(zip(self.variables, grads)):
            self._m[i] = self.beta_1 * self._m[i] + (1 - self.beta_1) * grad
            self._v[i] = self.beta_2 * self._v[i] + (1 - self.beta_2) * (
                grad**2
            )
            m_hat = self._m[i] / bias_correction_1
            v_hat = self._v[i] / bias_correction_2
            var.value = var.value - self.learning_rate * m_hat / (
                jnp.sqrt(v_hat) + self.epsilon
            )


def _train_step_tensorflow(
    model: "Model",
    n: int,
    flipout: bool = False,
    eager: bool = False,
    n_mc: int = 1,
) -> Any:
    """Get the training step function for TensorFlow."""
    import tensorflow as tf

    def train_fn(x_data, y_data):
        model.reset_kl_loss()
        with Sampling(n=n_mc, flipout=flipout):
            with tf.GradientTape() as tape:
                elbo_loss = model.elbo_loss(x_data, y_data, n, n_mc)
            variables = model.trainable_variables
            gradients = tape.gradient(elbo_loss, variables)
            model._optimizer.apply_gradients(zip(gradients, variables))
        return elbo_loss

    if eager:
        return train_fn
    else:
        return tf.function(train_fn)


def _train_step_pytorch(
    model: "Model",
    n: int,
    flipout: bool = False,
    eager: bool = False,
    n_mc: int = 1,
) -> Any:
    """Get the training step function for PyTorch."""
    import torch

    if eager:

        def train_fn(x_data, y_data):
            model.reset_kl_loss()
            with Sampling(n=n_mc, flipout=flipout):
                model._optimizer.zero_grad()
                elbo_loss = model.elbo_loss(x_data, y_data, n, n_mc)
                elbo_loss.backward()
                model._optimizer.step()
            return elbo_loss

        return train_fn

    # Use PyTorch tracing, for which we have to build a module,
    # and also a caching class for inputs of different sizes, b/c
    # last batch might have different number of datapoints
    else:

        class PyTorchModule(torch.nn.Module):
            def __init__(self, model: "Model"):
                super().__init__()
                for i, p in enumerate(model.trainable_variables):
                    setattr(self, str(i), p)
                self._probflow_model: Model = model

            def elbo_loss(self, *args) -> Any:
                self._probflow_model.reset_kl_loss()
                with Sampling(n=n_mc, flipout=flipout):
                    if len(args) == 1:
                        elbo_loss = self._probflow_model.elbo_loss(
                            None, args[0], n, n_mc
                        )
                    else:
                        elbo_loss = self._probflow_model.elbo_loss(
                            args[0], args[1], n, n_mc
                        )
                return elbo_loss

        class TraceCacher:
            """Cache traces for inputs of different sizes."""

            def __init__(self, model: "Model"):
                self.fns: dict[
                    str, Any
                ] = {}  # map from input shapes to traced function
                self.model: Model = model

            def get_traced_module(self, *args) -> Any:
                shape = "_".join(str(e.shape) for e in args)
                if shape in self.fns:
                    return self.fns[shape]
                else:
                    m = PyTorchModule(self.model)
                    inputs = {"elbo_loss": args}
                    self.fns[shape] = torch.jit.trace_module(
                        m, inputs, check_trace=False
                    )
                    return self.fns[shape]

            def __call__(self, *args) -> Any:
                self.model._optimizer.zero_grad()
                traced_module = self.get_traced_module(*args)
                elbo_loss = traced_module.elbo_loss(*args)
                elbo_loss.backward()
                self.model._optimizer.step()
                return elbo_loss

        pytorch_trainer = TraceCacher(model)

        def train_fn(x_data, y_data):
            if x_data is None:
                elbo_loss = pytorch_trainer(torch.tensor(y_data))
            else:
                elbo_loss = pytorch_trainer(
                    torch.tensor(x_data),
                    torch.tensor(y_data),
                )
            return elbo_loss

        return train_fn


def _train_step_jax(
    model: "Model",
    n: int,
    flipout: bool = False,
    eager: bool = False,
    n_mc: int = 1,
) -> Any:
    """Get the training step function for JAX."""
    import jax

    from probflow.utils.settings import _next_jax_key, jax_key_scope

    variables = model.trainable_variables

    def loss_fn(values: list[Any], key: Any, x_data: Any, y_data: Any) -> Any:
        for var, val in zip(variables, values):
            var.value = val
        model.reset_kl_loss()
        with jax_key_scope(key), Sampling(n=n_mc, flipout=flipout):
            return model.elbo_loss(x_data, y_data, n, n_mc)

    grad_fn = jax.value_and_grad(loss_fn)
    if not eager:
        grad_fn = jax.jit(grad_fn)

    def train_fn(x_data: Any, y_data: Any) -> Any:
        values = [v.value for v in variables]
        key = _next_jax_key()
        loss, grads = grad_fn(values, key, x_data, y_data)
        # Restore concrete values (loss_fn's mutation left stale trace-time
        # tracers on the variables, which must not escape the transformation)
        for var, val in zip(variables, values):
            var.value = val
        model._optimizer.step(grads)
        return loss

    return train_fn


def get_jax_sample_step(
    model: "Model", op_kind: str, n_arg: int | None = None
) -> Any:
    """Get a cached, jax.jit-compiled function for one predictive/sampling op.

    JAX has much higher per-operation overhead than TensorFlow/PyTorch when
    run eagerly, so ``predictive_sample``/``aleatoric_sample``/``epistemic_sample``
    are jitted here instead. Compiled functions are cached in the model
    instance (_jax_sample_fn_cache attr), keyed by the op kind, sampling settings,
    and the input shape (like the PyTorch ``TraceCacher``), so that repeated calls
    reuse the compiled program rather than recompiling eagerly.
    """
    import jax

    from probflow.utils.settings import (
        _next_jax_key,
        get_flipout,
        get_samples,
        jax_key_scope,
    )

    if not hasattr(model, "_jax_sample_fn_cache"):
        model._jax_sample_fn_cache = {}
    cache: dict[tuple, Any] = model._jax_sample_fn_cache
    variables = model.trainable_variables

    def apply_op(dist: Any) -> Any:
        if op_kind == "sample":
            return dist.sample()
        elif op_kind == "sample_n":
            return dist.sample(n=n_arg)
        elif op_kind == "mean":
            return dist.mean()
        else:
            raise ValueError(f"unknown jax sample op kind: {op_kind}")

    def build(has_x: bool) -> Any:
        def step_fn(values: list[Any], key: Any, x_data: Any) -> Any:
            for var, val in zip(variables, values):
                var.value = val
            with jax_key_scope(key):
                dist = model(x_data) if has_x else model()
                return apply_op(dist)

        return jax.jit(step_fn)

    def sample_fn(x_data: Any) -> Any:
        has_x = x_data is not None
        shape = None if x_data is None else tuple(x_data.shape)
        cache_key = (
            op_kind,
            n_arg,
            get_samples(),
            get_flipout(),
            has_x,
            shape,
        )
        if cache_key not in cache:
            cache[cache_key] = build(has_x)
        values = [v.value for v in variables]
        key = _next_jax_key()
        result = cache[cache_key](values, key, x_data)
        # Restore concrete values (step_fn's mutation left stale trace-time
        # tracers on the variables, which must not escape the transformation)
        for var, val in zip(variables, values):
            var.value = val
        return result

    return sample_fn


def get_training_step_function(
    model: "Model",
    n: int,
    flipout: bool = False,
    eager: bool = False,
    n_mc: int = 1,
):
    """Get the appropriate training step function for the current backend."""
    if get_backend() == ProbflowBackend.PYTORCH:
        return _train_step_pytorch(
            model=model, n=n, flipout=flipout, eager=eager, n_mc=n_mc
        )
    elif get_backend() == ProbflowBackend.JAX:
        return _train_step_jax(
            model=model, n=n, flipout=flipout, eager=eager, n_mc=n_mc
        )
    else:
        return _train_step_tensorflow(
            model=model, n=n, flipout=flipout, eager=eager, n_mc=n_mc
        )


def get_default_optimizer(
    trainable_variables: list[BackendVariable],
    learning_rate: float,
    **optimizer_kwargs,
):
    """Return the default optimizer for the current backend."""
    backend: ProbflowBackend = get_backend()
    if backend == ProbflowBackend.PYTORCH:
        import torch

        return torch.optim.Adam(
            trainable_variables,
            lr=learning_rate,
            **optimizer_kwargs,
        )
    elif backend == ProbflowBackend.TENSORFLOW:
        import tensorflow as tf

        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate, **optimizer_kwargs
        )
    elif backend == ProbflowBackend.JAX:
        return JaxAdam(
            trainable_variables,
            learning_rate,
            **optimizer_kwargs,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
