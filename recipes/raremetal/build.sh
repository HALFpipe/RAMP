#!/bin/bash

cget ignore zlib

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
    -DBUILD_TESTS="1" \
    ..

make
make test

install raremetal raremetalworker ${PREFIX}/bin

popd
