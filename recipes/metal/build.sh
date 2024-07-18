#!/bin/bash

set -x
set -e

# Use add_compile_options instead of set so that the compile options
# set by conda-build are not overwritten
sed -i CMakeLists.txt -e 's/set(CMAKE_CXX_FLAGS /add_compile_options(/'

sed -i metal/CMakeLists.txt -e 's#{CMAKE_BINARY_DIR}/bin#{CMAKE_INSTALL_BINDIR}#'

mkdir build
pushd build || exit

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_INSTALL_PREFIX:PATH="${PREFIX}" \
    ..

make
make test
make install
