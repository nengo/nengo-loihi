#!/usr/bin/env bash
set -e -v  # exit immediately on error, and print each line as it executes

# This script runs the test suite on the Loihi hardware (on the INRC cloud)

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    pip install coverage  # needed for the uploading in after_success
    openssl aes-256-cbc -K $encrypted_e0365add3e93_key -iv $encrypted_e0365add3e93_iv -in .ci/travis_rsa.enc -out ~/.ssh/id_rsa -d  # decrypt private key, see https://docs.travis-ci.com/user/encrypting-files/
    chmod 600 ~/.ssh/id_rsa
    echo -e "$INTELHOST_INFO" >> ~/.ssh/config  # ssh config, stored in travis-ci settings
    ssh -o StrictHostKeyChecking=no loihihost "echo 'Connected to loihihost'"
    scp -r . loihihost:/tmp/nengo-loihi-$TRAVIS_JOB_NUMBER
elif [[ "$COMMAND" == "script" ]]; then
    ssh loihihost << EOF
        sh /etc/profile
        sh ~/.bashrc
        cd /tmp/nengo-loihi-$TRAVIS_JOB_NUMBER
        conda create -y -n travis-ci-$TRAVIS_JOB_NUMBER python=3.5.2 scipy
        source activate travis-ci-$TRAVIS_JOB_NUMBER
        pip install -e .[tests]
        pip install ~/travis-ci/nxsdk-0.8.0.tar.gz
        SLURM=1 coverage run -m pytest --target loihi --no-hang -v --durations 50 --color=yes && \
        coverage report -m && \
        coverage xml
EOF
elif [[ "$COMMAND" == "after_success" ]]; then
    scp loihihost:/tmp/nengo-loihi-$TRAVIS_JOB_NUMBER/coverage.xml coverage.xml
    eval "bash <(curl -s https://codecov.io/bash)"
elif [[ "$COMMAND" == "after_script" ]]; then
    ssh loihihost "conda-env remove -y -n travis-ci-$TRAVIS_JOB_NUMBER"
else
    echo "$NAME does not define $COMMAND"
fi
