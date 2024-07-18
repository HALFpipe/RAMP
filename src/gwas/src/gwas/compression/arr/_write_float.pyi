from typing import Iterator

import numpy as np

def write_float(
    row_prefix_iterator: Iterator[bytes | None],
    data_array: np.ndarray,
    file_descriptor: int,
) -> None: ...
