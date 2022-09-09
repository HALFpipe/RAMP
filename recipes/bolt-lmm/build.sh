#!/bin/bash

cd src

make --jobs=$(nproc)

mkdir -p ${PREFIX}/bin
install -m775 bolt ${PREFIX}/bin/
