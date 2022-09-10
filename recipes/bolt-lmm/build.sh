#!/bin/bash

cd src

# make sure we don't disable assertions
export CPPFLAGS=$(
    echo "$CPPFLAGS" | sed \
    -e 's/-DNDEBUG//g'
)

# strip -march, -mtune and optimization flags
export CFLAGS=$(
    echo "$CFLAGS" | sed \
    -e 's/-O[^ ]\+//' -e 's/-march=[^ ]\+//' -e 's/-mtune=[^ ]\+//'
)
export CXXFLAGS=$(
    echo "$CXXFLAGS" | sed \
    -e 's/-O[^ ]\+//' -e 's/-march=[^ ]\+//' -e 's/-mtune=[^ ]\+//'
)

# add march back in
export CXXFLAGS="${CXXFLAGS} -march=ivybridge"

make --jobs=$(nproc)

mkdir -p ${PREFIX}/bin
install -m775 bolt ${PREFIX}/bin/
