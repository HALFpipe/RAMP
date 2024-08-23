from typing import Any, Iterable, Sequence

import numpy as np
from numpy import typing as npt


def invert_pivot(pivot: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """Calculates the inverse of a pivot array. Taken from
    https://stackoverflow.com/a/25535723

    Args:
        pivot (npt.NDArray[np.integer]): The pivot array.

    Returns:
        npt.NDArray[np.integer]: The inverse.
    """
    inverse = np.empty_like(pivot)
    inverse[pivot] = np.arange(pivot.size)
    return inverse


def scale_rows(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return (a.transpose() * b.ravel()).transpose()


def to_str(x: Any) -> str:
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
    if np.issubdtype(type(x), np.floating):
        if np.isnan(x):
            return "NA"
        return np.format_float_scientific(x)
    return str(x)


def make_sample_boolean_vectors(
    base_samples: Sequence[str],
    samples_iterable: Iterable[list[str]],
) -> list[npt.NDArray[np.bool_]]:
    return [
        np.fromiter((sample in samples for sample in base_samples), dtype=np.bool_)
        for samples in samples_iterable
    ]
