#!/usr/bin/env bash

shopt -s globstar

NAME=$0
COMMAND=$1
STATUS=0  # Used to exit with non-zero status if any check fails

function usage {
    echo "usage: $NAME <command>"
    echo
    echo "  install  Install dependencies for running static checks"
    echo "  run      Run static checks"
    exit 1
}

if [[ "$COMMAND" == "install" ]]; then
    conda install --quiet jupyter
    pip install codespell flake8 pylint
elif [[ "$COMMAND" == "run" ]]; then
    # Convert notebooks to Python scripts
    jupyter-nbconvert \
        --log-level WARN \
        --to python \
        --TemplateExporter.exclude_input_prompt=True \
        -- **/*.ipynb
    sed -i -e 's/# $/#/g' -e '/get_ipython()/d' -- docs/**/*.py
    flake8 nengo_loihi || STATUS=1
    flake8 --ignore=E226,E703,W291,W391 docs || STATUS=1
    pylint docs nengo_loihi || STATUS=1
    codespell -q 3 --skip="./build,./docs/_build,*-checkpoint.ipynb"|| STATUS=1
else
    if [[ -z "$COMMAND" ]]; then
        echo "Command required"
    else
        echo "Command $COMMAND not recognized"
    fi
    echo
    usage
fi
exit $STATUS
