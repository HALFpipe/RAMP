#!/bin/bash

export BOOST_LIB=${PREFIX}/include/boost
export EIGEN3_INCLUDE_DIR=${PREFIX}/include/eigen3
export MKLROOT=${PREFIX}
export SPECTRA_LIB=$(pwd)/spectra/include
export LIBRARY_PATH=${PREFIX}/lib

mkdir build
pushd build

cmake ../gcta

make -j$(nproc)

install gcta64 ${PREFIX}/bin

popd
