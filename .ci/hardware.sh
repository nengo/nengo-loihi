#!/usr/bin/env bash
if [[ ! -e .ci/common.sh || ! -e nengo_loihi ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script runs the test suite on the Loihi hardware (on the INRC cloud)

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    exe pip install coverage  # needed for the uploading in after_success
    openssl aes-256-cbc -K "${encrypted_e0365add3e93_key:?}" -iv "${encrypted_e0365add3e93_iv:?}" -in .ci/travis_rsa.enc -out ~/.ssh/id_rsa -d  # decrypt private key, see https://docs.travis-ci.com/user/encrypting-files/
    chmod 600 ~/.ssh/id_rsa
    echo -e "$INTELHOST_INFO" >> ~/.ssh/config  # ssh config, stored in travis-ci settings
    ssh -o StrictHostKeyChecking=no loihihost "echo 'Connected to loihihost'"
    exe scp -r . "loihihost:/tmp/nengo-loihi-$TRAVIS_JOB_NUMBER"
elif [[ "$COMMAND" == "script" ]]; then
    exe ssh loihihost << EOF
        sh /etc/profile
        sh ~/.bashrc
        HW_STATUS=0
        cd /tmp/nengo-loihi-$TRAVIS_JOB_NUMBER
        conda create -y -n travis-ci-$TRAVIS_JOB_NUMBER python=3.5.2 scipy
        source activate travis-ci-$TRAVIS_JOB_NUMBER
        pip install nengo-dl
        pip install -e .[tests]
        pip install $NENGO_VERSION
        pip install ~/travis-ci/nxsdk-0.8.0.tar.gz
        SLURM=1 coverage run -m pytest --target loihi --no-hang -v --durations 50 --color=yes -n 2 || HW_STATUS=1
        coverage report -m
        coverage xml
        exit \$HW_STATUS
EOF
elif [[ "$COMMAND" == "after_script" ]]; then
    exe scp "loihihost:/tmp/nengo-loihi-$TRAVIS_JOB_NUMBER/coverage.xml" coverage.xml
    eval "bash <(curl -s https://codecov.io/bash)"
    exe ssh loihihost "conda-env remove -y -n travis-ci-$TRAVIS_JOB_NUMBER"
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

exit "$STATUS"
