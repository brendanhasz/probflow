import pytest


def pytest_addoption(parser):
    """Provide pytest addoption."""
    parser.addoption(
        "--backend",
        action="store",
        default="tensorflow",
        help="run tests for a specific backend (tensorflow or pytorch)",
    )


@pytest.fixture(autouse=True)
def set_backend(request):
    """Provide set backend."""
    backend = request.config.getoption("--backend")
    import probflow as pf

    if backend not in [b.value for b in pf.ProbflowBackend]:
        raise ValueError(
            "Invalid backend specified. Must be 'tensorflow' or 'pytorch'."
        )

    pf.set_backend(pf.ProbflowBackend(backend))
