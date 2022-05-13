#!/bin/bash

mkdir build
pushd build

cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE="Release" \
    ..

make
make install

popd
