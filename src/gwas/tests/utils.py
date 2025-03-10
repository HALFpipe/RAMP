import tracemalloc
from contextlib import contextmanager
from functools import partial
from typing import Any

import numpy as np
import scipy
from jaxtyping import jaxtyped
from numpy import typing as npt
from typeguard import typechecked as typechecker

from gwas.log import logger

check_types: Any = partial(jaxtyped, typechecker=typechecker)


def regress(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    indices: npt.NDArray[np.bool_] | None = None,
) -> tuple[float, float, float, float]:
    if indices is None:
        a = a.ravel()
        b = b.ravel()
    else:
        a = a[indices]
        b = b[indices]

    missing_overlap = 1.0
    if np.isnan(a).any() or np.isnan(b).any():
        missing_overlap = scipy.spatial.distance.dice(np.isnan(a), np.isnan(b))

    mask = np.logical_and(np.isfinite(a), np.isfinite(b))
    a = a[mask]
    b = b[mask]

    a = a[:, np.newaxis]
    b = b[:, np.newaxis]

    x = np.hstack([np.ones_like(a), a])

    (intercept, slope), sum_residuals, _, _ = np.linalg.lstsq(x, b, rcond=None)

    if sum_residuals.size == 0:
        sum_residuals = np.array(np.inf)
    else:
        sum_residuals = sum_residuals[0]
    mean_residuals = sum_residuals.item() / a.size

    return float(intercept.item()), float(slope.item()), mean_residuals, missing_overlap


def check_bias(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    indices: npt.NDArray[np.bool_] | None = None,
    tolerance: float = 5e-2,
    check_slope: bool = True,
    check_residuals: bool = True,
) -> bool:
    intercept, slope, mean_residuals, missing_overlap = regress(a, b, indices)
    logger.debug(
        f"intercept={intercept:f} "
        f"|slope - 1|={np.abs(1 - slope):f} "
        f"mean_residuals={mean_residuals:f} "
        f"missing_overlap={missing_overlap:f}"
    )
    is_ok: np.bool_ | bool = True
    is_ok = is_ok and np.isclose(intercept, 0, atol=tolerance, rtol=tolerance)
    if check_slope:
        is_ok = is_ok and np.isclose(slope, 1, atol=tolerance, rtol=tolerance)
    if check_residuals:
        is_ok = is_ok and np.isclose(mean_residuals, 0, atol=tolerance, rtol=tolerance)
    return bool(is_ok)


def assert_both_close(
    genetic_variance: float,
    rmw_genetic_variance: float,
    error_variance: float,
    rmw_error_variance: float,
    atol=1e-3,
    rtol=1e-3,
) -> None:
    scale = max(
        abs(genetic_variance),
        abs(rmw_genetic_variance),
        abs(error_variance),
        abs(rmw_error_variance),
    )
    criterion = atol + rtol * scale
    assert np.abs(genetic_variance - rmw_genetic_variance) <= criterion
    assert np.abs(error_variance - rmw_error_variance) <= (atol + rtol * scale)


@contextmanager
def check_memory_leaks(target: int = 0):
    tracemalloc.clear_traces()
    tracemalloc.start()

    yield

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Ensure that we did not leak memory
    domain = np.lib.tracemalloc_domain
    domain_filter = tracemalloc.DomainFilter(inclusive=True, domain=domain)
    snapshot = snapshot.filter_traces([domain_filter])

    size = 0
    for trace in snapshot.traces:
        size += trace.size
        traceback = trace.traceback
        logger.info(f"allocation size {trace.size}: {'\n'.join(traceback.format())}")

    tracemalloc.clear_traces()

    assert size <= target
