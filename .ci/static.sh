#!/usr/bin/env bash
set -v  # print each line as it executes
shopt -s globstar

# This script runs the static style checks

NAME=$0
COMMAND=$1
STATUS=0  # used to exit with non-zero status if any check fails

if [[ "$COMMAND" == "install" ]]; then
    # pip installs a more recent entrypoints version than conda
    pip install entrypoints
    conda install --quiet jupyter
    pip install codespell flake8 pylint gitlint
elif [[ "$COMMAND" == "script" ]]; then
    # Convert notebooks to Python scripts
    jupyter-nbconvert \
        --log-level WARN \
        --to python \
        --TemplateExporter.exclude_input_prompt=True \
        -- **/*.ipynb
    sed -i -e 's/# $/#/g' -e '/get_ipython()/d' -- docs/**/*.py
    flake8 nengo_loihi || STATUS=1
    flake8 --ignore=E226,E703,W291,W391,W503 docs || STATUS=1
    pylint docs nengo_loihi || STATUS=1
    codespell -q 3 --skip="./build,./docs/_build,*-checkpoint.ipynb" || STATUS=1
    # undo single-branch cloning
    git config --replace-all remote.origin.fetch +refs/heads/*:refs/remotes/origin/*
    git fetch origin master
    N_COMMITS=$(git rev-list --count HEAD ^origin/master)
    for ((i=0; i<N_COMMITS; i++)) do
        git log -n 1 --skip $i --pretty=%B | gitlint -vvv || STATUS=1
    done
else
    echo "$NAME does not define $COMMAND"
fi
exit $STATUS
