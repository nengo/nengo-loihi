#!/usr/bin/env bash
set -e  # exit immediately on error
if [[ ! -e .ci/common.sh || ! -e nengo_loihi ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script builds the documentation and uploads it to GitHub pages

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    exe conda install --quiet numpy "pandoc<2"
    exe pip install cython jupyter matplotlib pillow requests scipy nengo-dl
    exe pip install "git+https://github.com/abr/abr_control.git"
    exe pip install -e .[docs]
elif [[ "$COMMAND" == "script" ]]; then
    exe sphinx-build -b linkcheck -v -W -D nbsphinx_execute=never docs docs/_build

    git clone -b gh-pages-release https://github.com/nengo/nengo-loihi.git ../nengo-docs
    RELEASES=$(find ../nengo-docs -maxdepth 1 -type d -name "v[0-9].*" -printf "%f,")

    if [[ "$TRAVIS_BRANCH" == "$TRAVIS_TAG" ]]; then
        RELEASES="$RELEASES$TRAVIS_TAG"
        exe sphinx-build -b html docs ../nengo-docs/"$TRAVIS_TAG" -vW -A building_version="$TRAVIS_TAG" -A releases="$RELEASES"
    else
        exe sphinx-build -b html docs ../nengo-docs -vW -A building_version=latest -A releases="$RELEASES"
    fi
elif [[ "$COMMAND" == "after_success" ]]; then
    cd ../nengo-docs
    git config --global user.email "travis@travis-ci.org"
    git config --global user.name "TravisCI"
    git add --all
    if [[ "$TRAVIS_BRANCH" == "$TRAVIS_TAG" ]]; then
        exe git commit -m "Documentation for release $TRAVIS_TAG"
        exe git push -q "https://$GH_TOKEN@github.com/nengo/nengo-loihi.git" gh-pages-release
    elif [[ "${TRAVIS_PULL_REQUEST_BRANCH:-$TRAVIS_BRANCH}" == "master" ]]; then
        exe git commit -m "Last update at $(date '+%Y-%m-%d %T')"
        exe git push -fq "https://$GH_TOKEN@github.com/nengo/nengo-loihi.git" gh-pages-release:gh-pages
    elif [[ "$TRAVIS_PULL_REQUEST" == "false" ]]; then
        exe git commit -m "Documentation for branch $TRAVIS_BRANCH"
        exe git push -fq "https://$GH_TOKEN@github.com/nengo/nengo-loihi.git" gh-pages-release:gh-pages-test
    fi
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

set +e  # reset options in case this is sourced
