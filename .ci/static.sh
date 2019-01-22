#!/usr/bin/env bash
set -e -v  # exit immediately on error, and print each line as it executes
shopt -s globstar

# This script runs the static style checks

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    conda install --quiet jupyter
    pip install codespell flake8 pylint
elif [[ "$COMMAND" == "script" ]]; then
    # Convert notebooks to Python scripts
    jupyter-nbconvert \
        --log-level WARN \
        --to python \
        --TemplateExporter.exclude_input_prompt=True \
        -- **/*.ipynb
    sed -i -e 's/# $/#/g' -e '/get_ipython()/d' -- docs/**/*.py
    flake8 nengo_loihi
    flake8 --ignore=E226,E703,W291,W391,W503 docs
    pylint docs nengo_loihi
    codespell -q 3 --skip="./build,./docs/_build,*-checkpoint.ipynb"
else
    echo "$NAME does not define $COMMAND"
fi
