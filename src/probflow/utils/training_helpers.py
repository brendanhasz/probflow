"""Helper functions for backend-specific tasks during training."""
from typing import Any

from probflow.utils.typing import BackendVariable
from probflow.utils.settings import ProbflowBackend, Sampling, get_backend

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from probflow.models.model import Model

def _train_step_tensorflow(
    model: "Model", n: int, flipout: bool = False, eager: bool = False, n_mc: int = 1
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
    model: "Model", n: int, flipout: bool = False, eager: bool = False, n_mc: int = 1
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
                super(PyTorchModule, self).__init__()
                for i, p in enumerate(model.trainable_variables):
                    setattr(self, str(i), p)
                self._probflow_model: "Model" = model

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
                self.fns: dict[str, Any] = {}  # map from input shapes to traced function
                self.model: "Model" = model

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


def get_training_step_function(
    model: "Model", n: int, flipout: bool = False, eager: bool = False, n_mc: int = 1
):
    """Get the appropriate training step function for the current backend."""
    if get_backend() == ProbflowBackend.PYTORCH:
        return _train_step_pytorch(
            model=model, n=n, flipout=flipout, eager=eager, n_mc=n_mc
        )
    else:
        return _train_step_tensorflow(
            model=model, n=n, flipout=flipout, eager=eager, n_mc=n_mc
        )



def get_default_optimizer(trainable_variables: list[BackendVariable], learning_rate: float, **optimizer_kwargs):
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
    else:
        raise ValueError(f"Unsupported backend: {backend}")
