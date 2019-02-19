#!/usr/bin/env bash
shopt -s globstar
if [[ ! -e .ci/common.sh || ! -e nengo_loihi ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script runs the static style checks

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    # pip installs a more recent entrypoints version than conda
    exe pip install entrypoints
    exe conda install --quiet jupyter
    exe pip install codespell flake8 pylint gitlint
elif [[ "$COMMAND" == "script" ]]; then
    # Convert notebooks to Python scripts
    jupyter-nbconvert \
        --log-level WARN \
        --to python \
        --TemplateExporter.exclude_input_prompt=True \
        -- **/*.ipynb
    sed -i -e 's/# $/#/g' -e '/get_ipython()/d' -- docs/**/*.py
    exe flake8 nengo_loihi
    exe flake8 --ignore=E226,E703,W291,W391,W503 docs
    exe pylint docs nengo_loihi
    exe codespell -q 3 --skip="./build,./docs/_build,*-checkpoint.ipynb"
    exe shellcheck -e SC2087 .ci/*.sh
    # undo single-branch cloning
    git config --replace-all remote.origin.fetch +refs/heads/*:refs/remotes/origin/*
    git fetch origin master
    N_COMMITS=$(git rev-list --count HEAD ^origin/master)
    for ((i=0; i<N_COMMITS; i++)) do
        git log -n 1 --skip $i --pretty=%B | grep -v '^Co-authored-by:' | exe gitlint -vvv
    done
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

exit "$STATUS"
