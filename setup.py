#!/usr/bin/env python
import imp
import io
import os
import sys

from setuptools import find_packages, setup


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    "version", os.path.join(root, "nengo_loihi", "version.py"))
testing = "test" in sys.argv or "pytest" in sys.argv

install_requires = [
    "jinja2",
    "nengo>=2.8.0",
]
docs_require = [
    "abr_control @ git+https://github.com/abr/abr_control",
    "jupyter",
    "matplotlib>=2.0",
    "nbsphinx",
    "nbconvert",
    "nengo-dl>=2.1.1",
    "nengo-extras",
    "nengo_sphinx_theme>=0.7",
    "numpydoc>=0.6",
    "sphinx>=1.8",
]
tests_require = [
    "coverage>=4.3",
    "nengo-extras",
    "pytest>=3.4,<4",
    "pytest-cov>=2.6.0",
    # pytest-xdist>=1.28 requires pytest>=4.4, which nengo<3 doesn't support
    "pytest-xdist>=1.26.0,<1.28.0",
    "matplotlib>=2.0",
    "scipy",
]

setup(
    name="nengo_loihi",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/nengo/nengo-loihi",
    license="Free for non-commercial use",
    description="Run Nengo models on the Loihi chip",
    long_description=read("README.rst"),
    zip_safe=False,
    python_requires=">=3.4",
    package_data={"nengo_loihi": ["nengo_loihi/snips/*"]},
    include_package_data=True,
    setup_requires=[
        "nengo",
    ],
    install_requires=install_requires,
    extras_require={
        "all": docs_require + tests_require,
        "docs": docs_require,
        "tests": tests_require,
    },
    entry_points={
        'nengo.backends': [
            'loihi = nengo_loihi:Simulator'
        ],
    },
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
