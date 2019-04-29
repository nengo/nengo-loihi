#!/usr/bin/env python

# Automatically generated by nengo-bones, do not edit this file directly
# Version: 0.2.0.dev0

import io
import os
import runpy

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version = runpy.run_path(os.path.join(
    root, "nengo_loihi", "version.py"))["version"]

install_req = [
    "jinja2",
    "nengo>=2.8.0",
]
doc_req = [
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
optional_req = [
]
test_req = [
    "coverage>=4.3",
    "nengo-extras",
    "pytest>=3.4,<4",
    "pytest-cov>=2.6.0",
    "pytest-xdist>=1.26.0,<1.28.0",
    "matplotlib>=2.0",
    "scipy",
]

setup(
    name="nengo-loihi",
    version=version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    url="https://www.nengo.ai/nengo-loihi",
    include_package_data=True,
    license="Free for non-commercial use",
    description="Run Nengo models on the Loihi chip",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    install_requires=install_req,
    extras_require={
        "all": doc_req + optional_req + test_req,
        "doc": doc_req,
        "optional": optional_req,
        "test": test_req,
    },
    python_requires=">=3.4",
    entry_points={
        "nengo.backends": [
            "loihi = nengo_loihi:Simulator",
        ],
    },
    package_data={
        "nengo_loihi": [
            "nengo_loihi/snips/*",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: Nengo",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
