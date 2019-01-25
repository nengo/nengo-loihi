#!/usr/bin/env bash
set -e -v  # exit immediately on error, and print each line as it executes

# This script builds the documentation and uploads it to GitHub pages

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    conda install --quiet numpy "pandoc<2"
    pip install cython jupyter matplotlib pillow requests scipy
    pip install "git+https://github.com/nengo/nengo-dl.git@conv_transform"
    pip install "git+https://github.com/abr/abr_control.git"
    pip install -e .[docs]
elif [[ "$COMMAND" == "script" ]]; then
    sphinx-build -b linkcheck -v -W -D nbsphinx_execute=never docs docs/_build

    git clone -b gh-pages-release https://github.com/nengo/nengo-loihi.git ../nengo-docs
    RELEASES=$(find ../nengo-docs -maxdepth 1 -type d -name "v[0-9].*" -printf "%f,")

    if [[ "$TRAVIS_BRANCH" == "$TRAVIS_TAG" ]]; then
        RELEASES="$RELEASES$TRAVIS_TAG"
        sphinx-build -b html docs ../nengo-docs/"$TRAVIS_TAG" -vW -A building_version="$TRAVIS_TAG" -A releases="$RELEASES"
    else
        sphinx-build -b html docs ../nengo-docs -vW -A building_version=latest -A releases="$RELEASES"
    fi
elif [[ "$COMMAND" == "after_success" ]]; then
    cd ../nengo-docs
    git config --global user.email "travis@travis-ci.org"
    git config --global user.name "TravisCI"
    git add --all
    if [[ "$TRAVIS_BRANCH" == "$TRAVIS_TAG" ]]; then
        git commit -m "Documentation for release $TRAVIS_TAG"
        git push -q "https://$GH_TOKEN@github.com/nengo/nengo-loihi.git" gh-pages-release
    elif [[ "${TRAVIS_PULL_REQUEST_BRANCH:-$TRAVIS_BRANCH}" == "master" ]]; then
        git commit -m "Last update at $(date '+%Y-%m-%d %T')"
        git push -fq "https://$GH_TOKEN@github.com/nengo/nengo-loihi.git" gh-pages-release:gh-pages
    elif [[ "$TRAVIS_PULL_REQUEST" == "false" ]]; then
        git commit -m "Documentation for branch $TRAVIS_BRANCH"
        git push -fq "https://$GH_TOKEN@github.com/nengo/nengo-loihi.git" gh-pages-release:gh-pages-test
    fi
else
    echo "$NAME does not define $COMMAND"
fi
