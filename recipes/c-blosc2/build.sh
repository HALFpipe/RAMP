#!/bin/bash

rm -rf internal-complibs

mkdir build
pushd build || exit

cmake -G "Unix Makefiles" \
    ${CMAKE_ARGS} \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=1 \
    -DBUILD_STATIC=0 \
    -DBUILD_SHARED=1 \
    -DBUILD_TESTS=1 \
    -DBUILD_EXAMPLES=0 \
    -DBUILD_BENCHMARKS=0 \
    -DPREFER_EXTERNAL_LZ4=1 \
    -DPREFER_EXTERNAL_ZSTD=1 \
    -DPREFER_EXTERNAL_ZLIB=1 \
    "${SRC_DIR}"

cmake --build . --parallel "$(nproc)"
ctest
cmake --build . --target install

popd || exit
