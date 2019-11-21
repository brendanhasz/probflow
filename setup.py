from setuptools import find_packages, setup

with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name='probflow',
    version='2.0.0a2', #NOTE: also in __init__.py
    author='Brendan Hasz',
    author_email='winsto99@gmail.com',
    description='A Python package for building Bayesian models with TensorFlow or PyTorch',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/brendanhasz/probflow',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Development Status :: 3 - Alpha',
    ],
    zip_safe=False,
    extras_require={
        'tests': [
            'tensorflow == 2.0.0',
            'tensorflow-probability == 0.8.0',
            'torch >= 1.2.0',
            'pytest >= 5.1.2',
            'pytest-cov >= 2.7.1',
            'pylint >= 2.3.1',
            ],
        'docs': [
            'tensorflow == 2.0.0',
            'tensorflow-probability == 0.8.0',
            'sphinx-tabs'
            ]
    }
)
