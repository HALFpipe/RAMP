# -*- coding: utf-8 -*-

from multiprocessing import cpu_count
from pathlib import Path
from subprocess import check_call

import numpy as np
from numpy import typing as npt

from gwas.log import logger
from gwas.utils import unwrap_which

bcftools = unwrap_which("bcftools")
gcta64 = unwrap_which("gcta64")
plink2 = unwrap_which("plink2")
rmw = unwrap_which("raremetalworker")
tabix = unwrap_which("tabix")


def to_bgzip(base_path: Path, zstd_path: Path) -> Path:
    gz_path = base_path / f"{zstd_path.stem}.gz"

    if not gz_path.is_file():
        check_call(
            [
                "bash",
                "-c",
                f"zstd -c -d --long=31 {zstd_path} | "
                f"bgzip --threads {cpu_count()} > {gz_path}",
            ]
        )

    tbi_path = gz_path.with_suffix(".gz.tbi")
    if not tbi_path.is_file():
        check_call(
            [
                tabix,
                "-p",
                "vcf",
                str(gz_path),
            ]
        )

    return gz_path


def is_bfile(path: Path) -> bool:
    return all(
        (path.parent / f"{path.name}{suffix}").is_file()
        for suffix in {".bed", ".bim", ".fam"}
    )


def is_pfile(path: Path) -> bool:
    return all(
        (path.parent / f"{path.name}{suffix}").is_file()
        for suffix in {".pgen", ".pvar", ".psam"}
    )


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
        sum_residuals = sum_residuals[0]
    mean_residuals = sum_residuals / a.size

    return float(intercept), float(slope), float(mean_residuals)


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
