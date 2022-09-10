#!/bin/bash

# disable security features that are crashing the build
export CPPFLAGS=$(
    echo "$CPPFLAGS" | sed \
    -e 's/-D_FORTIFY_SOURCE=[^ ]\+//g'
)
export CPPFLAGS="${CPPFLAGS} -U_FORTIFY_SOURCE"
cget init --verbose \
    --cflags "-U_FORTIFY_SOURCE" \
    --cxxflags "-U_FORTIFY_SOURCE"

cget ignore --verbose zlib

# avoid race condition by monkey patching cpu_count to disable
# parallel builds within `cget`
python - <<EOF
import multiprocessing
multiprocessing.cpu_count = lambda: 1

from cget.prefix import CGetPrefix

prefix = CGetPrefix("cget", verbose=True)
for package_build in prefix.from_file("requirements.txt"):
    prefix.install(package_build)
EOF

mkdir build
pushd build

cmake \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_TOOLCHAIN_FILE="../cget/cget/cget.cmake" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ..

make

install DosageConvertor ${PREFIX}/bin

popd
