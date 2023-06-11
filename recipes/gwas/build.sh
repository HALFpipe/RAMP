#!/bin/bash

set -x

CFLAGS="-isystem ${CONDA_BUILD_SYSROOT}/usr/include "${CFLAGS}
LDFLAGS="-isystem ${CONDA_BUILD_SYSROOT}/usr/include "${LDFLAGS}
CPPFLAGS="-isystem ${CONDA_BUILD_SYSROOT}/usr/include "${CPPFLAGS}

CFLAGS=$(echo "${CFLAGS}" | sed "s/-O2/-O3/g")
CPPFLAGS=$(echo "${CPPFLAGS}" | sed "s/-O2/-O3/g")
CXXFLAGS=$(echo "${CXXFLAGS}" | sed "s/-O2/-O3/g")

export CPPFLAGS CFLAGS CXXFLAGS LDFLAGS

python -m pip install --no-deps --ignore-installed "./src/gwas"
