from functools import partial

import numpy as np
from gwas.log import logger
from jaxtyping import jaxtyped
from numpy import typing as npt
from typeguard import typechecked as typechecker

check_types = partial(jaxtyped, typechecker=typechecker)


def regress(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    indices: npt.NDArray[np.bool_] | None = None,
) -> tuple[float, float, float]:
    if indices is None:
        a = a.ravel()
        b = b.ravel()
    else:
        a = a[indices]
        b = b[indices]

    a = a[:, np.newaxis]
    b = b[:, np.newaxis]

    x = np.hstack([np.ones_like(a), a])

    sum_residuals: float | npt.NDArray[np.float64] = np.inf
    (intercept, slope), sum_residuals, _, _ = np.linalg.lstsq(x, b, rcond=None)

    if sum_residuals.size == 0:
        sum_residuals = np.inf
    else:
        sum_residuals = float(sum_residuals[0])
    mean_residuals = sum_residuals / a.size

    return float(intercept.item()), float(slope.item()), mean_residuals


def check_bias(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    indices: npt.NDArray[np.bool_] | None = None,
    tolerance: float = 5e-2,
    check_slope: bool = True,
    check_residuals: bool = True,
) -> bool:
    intercept, slope, mean_residuals = regress(a, b, indices)
    logger.debug(
        f"intercept={intercept:f} "
        f"|slope - 1|={np.abs(1 - slope):f} "
        f"mean_residuals={mean_residuals:f}"
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
