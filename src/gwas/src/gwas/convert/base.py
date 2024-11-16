import numpy as np

from ..compression.arr.base import (
    compression_methods,
)

dtype = np.float64
compression_method_name = "parquet"
compression_method = compression_methods[compression_method_name]
