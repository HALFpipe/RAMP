#!/bin/bash

set -x

export LDFLAGS="${LDFLAGS} -lrt -lsqlite3 -lzstd"

mkdir -p "${PREFIX}/bin"

./waf configure --verbose --prefix="${PREFIX}"

cat <<EOF >build/release/config/package_revision_autogenerated.hpp
#ifndef PACKAGE_REVISION_HPP
#define PACKAGE_REVISION_HPP
namespace globals {
char const* const package_version = "${PKG_VERSION}" ;
char const* const package_revision = "unknown" ;
}
#endif
EOF

./waf build --verbose --jobs="$(nproc)"

./waf install --verbose

pushd "${PREFIX}/bin" || exit
for i in *_v"${PKG_VERSION}"; do
    ln -s "${i}" "$(basename "${i}" "_v${PKG_VERSION}")"
done
popd || exit
