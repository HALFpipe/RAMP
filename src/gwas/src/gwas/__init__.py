# -*- coding: utf-8 -*-
import os
from importlib.metadata import PackageNotFoundError, version

from jax import config

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Show compression error messages.
os.environ["BLOSC_TRACE"] = "1"

config.update("jax_enable_x64", True)
del config
del os
