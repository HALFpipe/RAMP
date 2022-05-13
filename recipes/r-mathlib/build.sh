#!/bin/bash

./configure --prefix=${PREFIX} \
    --with-blas \
    --with-lapack \
    --without-cairo \
    --without-libpng \
    --without-libtiff \
    --without-jpeglib \
    --without-readline \
    --without-recommended-packages \
    --without-tcltk \
    --without-x

pushd src/nmath/standalone

make
make install

popd
