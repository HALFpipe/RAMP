#!/bin/bash

# Ensure that hard-coded executables exist.
mkdir -p "$(pwd)/bin"
prefix="${BUILD_PREFIX}/bin/${BUILD}-"
for tool in CC GXX AR; do
    executable=$(realpath ${!tool})
    tool_name=${executable#"${prefix}"}
    ln -s "${executable}" "$(pwd)/bin/${tool_name}"
done
PATH="$(pwd)/bin:$PATH"
export PATH

# Disable security features that are crashing the build.
CPPFLAGS=$(
    echo "$CPPFLAGS" | sed \
        -e 's/-D_FORTIFY_SOURCE=[^ ]\+//g'
)
CPPFLAGS="${CPPFLAGS} -U_FORTIFY_SOURCE"
export CPPFLAGS
cget init --verbose \
    --cflags "-U_FORTIFY_SOURCE" \
    --cxxflags "-U_FORTIFY_SOURCE"

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

# Actually build the project.
mkdir build
pushd build || exit
cmake \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_TOOLCHAIN_FILE="../cget/cget/cget.cmake" \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    ..

make

install DosageConvertor "${PREFIX}/bin"

popd || exit
