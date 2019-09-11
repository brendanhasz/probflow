from setuptools import find_packages, setup

setup(name='probflow',
    version='2.0.0',
    description='A Python package for building Bayesian models with TensorFlow or PyTorch',
    url='https://github.com/brendanhasz/probflow',
    author='Brendan Hasz',
    author_email='winsto99@gmail.com',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=False,
    extras_require={
        'tests': [
            'tensorflow == 2.0.0b1',
            'tfp-nightly >= 0.9.0',
            'torch >= 1.2.0',
            'pytest >= 5.1.2',
            'pylint >= 2.3.1',
            ],
        'docs': [
            'tensorflow == 2.0.0b1',
            'tfp-nightly >= 0.9.0',
            ]
    }
)
