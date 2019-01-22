#!/usr/bin/env bash
set -e -v  # exit immediately on error, and print each line as it executes

# This script sets up the conda environment for all the other scripts

NAME=$0
COMMAND=$1
MINICONDA="http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh"

if [[ "$COMMAND" == "install" ]]; then
    wget "$MINICONDA" --quiet -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
    conda info -a
    conda create -q -n test python="$PYTHON" pip
    source activate test
else
    echo "$NAME does not define $COMMAND"
fi
