#!/usr/bin/env python

# Automatically generated by nengo-bones, do not edit this file directly

import io
import pathlib
import runpy

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py"
    )


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = pathlib.Path(__file__).parent
version = runpy.run_path(str(root / "nengo_loihi" / "version.py"))["version"]

install_req = [
    "jinja2",
    "nengo>=3.1.0",
    "packaging",
    "scipy>=1.2.1",
]
docs_req = [
    "nengo_sphinx_theme>=0.7",
    "numpydoc>=0.6",
    "sphinx>=1.8",
]
optional_req = [
    "nengo-extras>=0.5",
    "networkx-metis>=1.0",
]
tests_req = [
    "coverage>=4.3",
    "nengo-extras>=0.5",
    "pytest>=5.0.0",
    "pytest-allclose>=1.0.0",
    "pytest-cov>=2.6.0",
    "pytest-plt>=1.0.0",
    "pytest-rng>=1.0.0",
    "pytest-xdist>=2.0.0",
    "matplotlib>=2.0",
]

setup(
    name="nengo-loihi",
    version=version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    url="https://www.nengo.ai/nengo-loihi",
    include_package_data=True,
    license="Apache 2.0 license",
    description="Run Nengo models on the Loihi chip",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    install_requires=install_req,
    extras_require={
        "all": docs_req + optional_req + tests_req,
        "docs": docs_req,
        "optional": optional_req,
        "tests": tests_req,
    },
    python_requires=">=3.6",
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
        "Development Status :: 4 - Beta",
        "Framework :: Nengo",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
