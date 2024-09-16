#!/bin/bash

set -x
set -e

# Use nodejs from build prefix
rm "${PREFIX}/bin/node"
ln -s "${BUILD_PREFIX}/bin/node" "${PREFIX}/bin/node"

# Don't use pre-built gyp packages
export npm_config_build_from_source=true

# Ignore custom configuration on build machine
export NPM_CONFIG_USERCONFIG=/tmp/nonexistentrc

npm clean-install
npm run build
packed=$(npm pack)
npm install --global "${packed}"
