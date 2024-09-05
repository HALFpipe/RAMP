from typing import Iterator, Sequence

import numpy as np

def write_float(
    row_prefix_iterator: Iterator[bytes | None],
    arrays: Sequence[np.ndarray],
    file_descriptor: int,
    num_threads: int,
) -> None: ...
