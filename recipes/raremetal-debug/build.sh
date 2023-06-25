#!/bin/bash

# Ensure that hard-coded executables exist.
mkdir -p "$(pwd)/bin"
conda_prefix="${BUILD_PREFIX}/bin/${BUILD}-"
for tool in GCC GXX AR; do
    executable=$(realpath ${!tool})
    tool_name=${executable#"${conda_prefix}"}
    ln -s "${executable}" "$(pwd)/bin/${tool_name}"
done
PATH="$(pwd)/bin:$PATH"
export PATH

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
pushd build || exit

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="../cget/cget/cget.cmake" \
    -DBUILD_TESTS:BOOL=ON \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ..

make
# Skip tests for debug build
# make test

mv -nv raremetal raremetal-debug
mv -nv raremetalworker raremetalworker-debug
install raremetal-debug raremetalworker-debug "${PREFIX}"/bin

popd || exit
