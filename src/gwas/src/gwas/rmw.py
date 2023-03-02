# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
from numpy import typing as npt

from gwas.z import CompressedTextReader

names = (
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "N_INFORMATIVE",
    "FOUNDER_AF",
    "ALL_AF",
    "INFORMATIVE_ALT_AC",
    "CALL_RATE",
    "HWE_PVALUE",
    "N_REF",
    "N_HET",
    "N_ALT",
    "U_STAT",
    "SQRT_V_STAT",
    "ALT_EFFSIZE",
    "PVALUE",
)


def read_scorefile(
    file_path: Path | str,
):
    line_count = 0
    with CompressedTextReader(file_path) as file:
        for line in file:
            if line.startswith("#"):
                continue
            line_count += 1

    formats = (
        object,
        int,
        object,
        object,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    )

    dtype = np.dtype(list(zip(names, formats)))

    array = np.empty((line_count,), dtype=dtype)

    j = 0
    with CompressedTextReader(file_path) as file:
        for line in file:
            if line.startswith("#"):
                continue
            tokens = line.split()
            token: float | str | None = None
            for i, token in enumerate(tokens):
                if token == "NA":
                    token = np.nan
                array[j][i] = token
            j += 1

    return array


def save_scorefile(scores: npt.NDArray):
    pass
