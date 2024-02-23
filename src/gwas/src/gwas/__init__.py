# -*- coding: utf-8 -*-
import os
from importlib.metadata import PackageNotFoundError, version

import torch

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

# Show compression error messages.
os.environ["BLOSC_TRACE"] = "1"
# os.environ["MKL_VERBOSE"] = "1"

torch.set_default_dtype(torch.float64)
del torch
del os
