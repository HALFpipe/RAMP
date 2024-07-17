# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import NamedTuple, Self

import numpy as np

# from ..log import logger
from ..utils import chromosome_to_int


class Variant(NamedTuple):
    chromosome_int: int
    position: int
    reference_allele: str
    alternate_allele: str

    is_imputed: bool
    alternate_allele_frequency: float
    minor_allele_frequency: float
    r_squared: float

    format_str: str

    @classmethod
    def from_metadata_columns(
        cls,
        chromosome_str: str,
        position_str: str,
        reference_allele: str,
        alternate_allele: str,
        info_str: str,
        format_str: str,
    ) -> Self:
        # logger.debug(
        #     f"Reading variant "
        #     f"chromosome_str: {chromosome_str}, position_str: {position_str}, "
        #     f"reference_allele: {reference_allele}, "
        #     f"alternate_allele: {alternate_allele}, info_str: {info_str}, "
        #     f"format_str: {format_str}"
        # )

        chromosome: int | str = chromosome_str
        if isinstance(chromosome, str) and chromosome.isdigit():
            chromosome = int(chromosome)

        position: int = int(position_str)

        info_tokens = info_str.split(";")
        info: dict[str, str] = dict()
        for token in info_tokens:
            if "=" not in token:
                continue
            token, value = token.split("=")
            info[token] = value

        is_imputed = "IMPUTED" in info_tokens
        alternate_allele_frequency = float(info.get("AF", np.nan))
        minor_allele_frequency = float(info.get("MAF", np.nan))
        r_squared = float(info.get("R2", np.nan))

        return cls(
            chromosome_to_int(chromosome),
            position,
            reference_allele,
            alternate_allele,
            is_imputed,
            alternate_allele_frequency,
            minor_allele_frequency,
            r_squared,
            format_str,
        )
