from setuptools import find_packages, setup

with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name="probflow",
    version="2.0.0a3",  # NOTE: also in __init__.py
    author="Brendan Hasz",
    author_email="winsto99@gmail.com",
    description="A Python package for building Bayesian models with TensorFlow or PyTorch",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/brendanhasz/probflow",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    zip_safe=False,
    extras_require={
        "tensorflow_cpu": [
            "tensorflow == 2.2.0",
            "tensorflow-probability == 0.10.0",
        ],
        "tensorflow_gpu": [
            "tensorflow-gpu == 2.2.0",
            "tensorflow-probability == 0.10.0",
        ],
        "pytorch": [
            "torch >= 1.5.0",
        ],
        "tests": [
            "tensorflow == 2.2.0",
            "tensorflow-probability == 0.10.0",
            "torch >= 1.5.0",
            "pytest >= 5.1.2",
            "pytest-cov >= 2.7.1",
            "flake8 >= 3.8.3",
            "black >= 19.10b0",
            "isort >= 5.1.2",
            "sphinx >= 3.1.2",
            "sphinx-tabs >= 1.1.13",
            "sphinx_rtd_theme >= 0.5.0",
            "setuptools >= 49.1.0",
            "wheel >= 0.34.2",
            "twine >= 3.2.0",
        ],
        "docs": [
            "tensorflow == 2.2.0",
            "tensorflow-probability == 0.10.0",
            "sphinx-tabs >= 1.1.13",
        ],
    },
)
