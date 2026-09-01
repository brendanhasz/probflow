import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="tensorflow",
        help="run tests for a specific backend (tensorflow or pytorch)",
    )


@pytest.fixture(autouse=True)
def set_backend(request):
    backend = request.config.getoption("--backend")
    if backend not in ["tensorflow", "pytorch"]:
        raise ValueError(
            "Invalid backend specified. Must be 'tensorflow' or 'pytorch'."
        )
    import probflow as pf

    pf.set_backend(backend)
