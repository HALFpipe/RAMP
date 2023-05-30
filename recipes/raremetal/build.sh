#!/bin/bash

# Ensure that hard-coded executables exist.
mkdir -p $(pwd)/bin
prefix="${BUILD_PREFIX}/bin/${BUILD}-"
for tool in GCC GXX AR; do
    executable=$(realpath ${!tool})
    tool_name=${executable#"${prefix}"}
    ln -s ${executable} $(pwd)/bin/${tool_name}
done
export PATH="$(pwd)/bin:$PATH"

# Use conda's compiler flags.
cget init --verbose \
    --cflags "${CFLAGS} -fopenmp -g" \
    --cxxflags "${CXXFLAGS} -fopenmp -g"

# Use conda's zlib.
cget ignore --verbose zlib

# Avoid race condition by monkey patching cpu_count to disable
# parallel builds within `cget`.
python - <<EOF
import multiprocessing
multiprocessing.cpu_count = lambda: 1

from cget.prefix import CGetPrefix

prefix = CGetPrefix("cget", verbose=True)
for package_build in prefix.from_file("requirements.txt"):
    prefix.install(package_build)
EOF

# Actually build `raremetal`.
mkdir build
pushd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="../cget/cget/cget.cmake" \
    -DBUILD_TESTS:BOOL=ON \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ..

make
make test

install raremetal raremetalworker ${PREFIX}/bin

popd
