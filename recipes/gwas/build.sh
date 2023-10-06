#!/bin/bash

set -x

CFLAGS=${CFLAGS//-O2/-O3}
CPPFLAGS=${CPPFLAGS//-O2/-O3}
CXXFLAGS=${CXXFLAGS//-O2/-O3}\

LD_LIBRARY_PATH=${PREFIX}/lib

export CPPFLAGS CFLAGS CXXFLAGS LDFLAGS LD_LIBRARY_PATH

python -m pip install --verbose --no-deps --ignore-installed "./src/gwas"
