#!/usr/bin/env bash
# Small snippets common to all CI scripts.
# All CI scripts should source this script.

exe() {
    echo "\$ $*";
    "$@";
}
