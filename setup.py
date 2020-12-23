from setuptools import find_packages, setup

with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name="probflow",
    version="2.1.0",
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
    install_requires=[
        "matplotlib >= 3.1.0",
        "numpy >= 1.17.0",
        "pandas >= 0.25.0",
        "cloudpickle >= 1.3",
    ],
    extras_require={
        "tensorflow": [
            "tensorflow >= 2.2.0",
            "tensorflow-probability >= 0.10.0",
        ],
        "tensorflow_gpu": [
            "tensorflow-gpu >= 2.2.0",
            "tensorflow-probability >= 0.10.0",
        ],
        "pytorch": [
            "torch >= 1.5.0",
        ],
        "dev": [
            "black >= 19.10b0",
            "bumpversion",
            "flake8 >= 3.8.3",
            "isort >= 5.1.2",
            "pytest >= 6.0.0rc1",
            "pytest-cov >= 2.7.1",
            "sphinx >= 3.1.2",
            "sphinx-tabs >= 1.1.13",
            "sphinx_rtd_theme >= 0.5.0",
            "setuptools >= 49.1.0",
            "twine >= 3.2.0",
            "wheel >= 0.34.2",
        ],
        "docs": [
            "tensorflow >= 2.2.0",
            "tensorflow-probability >= 0.10.0",
            "sphinx-tabs >= 1.1.13",
        ],
    },
)
