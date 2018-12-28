#!/usr/bin/env bash
set -e  # exit immediately on error
if [[ ! -e .ci/common.sh || ! -e nengo_loihi ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script runs the test suite on the emulator

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    exe conda install --quiet matplotlib mkl numpy scipy tensorflow
    exe pip install nengo-dl
    exe pip install -e ".[tests]"
elif [[ "$COMMAND" == "script" ]]; then
    exe coverage run -m pytest nengo_loihi -v --duration 20 --plots --color=yes
    exe coverage run -a -m pytest --pyargs nengo -v --duration 20 --color=yes
    exe coverage report -m
elif [[ "$COMMAND" == "after_success" ]]; then
    eval "bash <(curl -s https://codecov.io/bash)"
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

set +e  # reset options in case this is sourced
