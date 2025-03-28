from dataclasses import dataclass
from types import TracebackType
from typing import Type

import numpy as np
from numpy import typing as npt

from ..utils.genetics import chromosome_to_int
from ._htslib import read_dosages, read_variants
from .base import Engine, VCFFile
from .variant import Variant


def variant_from_htslib(
    chromosome_str: str,
    position: int,
    reference_allele: str,
    alternate_allele: str,
    is_imputed: bool,
    alternate_allele_frequency: float,
    minor_allele_frequency: float,
    r2_value: float,
) -> Variant:
    return Variant(
        chromosome_to_int(chromosome_str),
        position,
        reference_allele,
        alternate_allele,
        is_imputed,
        alternate_allele_frequency,
        minor_allele_frequency,
        r2_value,
    )


@dataclass
class HtslibVCFFile(VCFFile):
    engine: Engine = Engine.htslib

    def __post_init__(self) -> None:
        variants, samples = read_variants(str(self.file_path), variant_from_htslib)

        self.vcf_samples = samples
        self.set_samples(set(samples))

        self.shared_vcf_variants = self.make_shared_data_frame(variants, self.sw)
        self.variant_indices = np.arange(self.vcf_variant_count, dtype=np.uint32)
        self.update_chromosome()

    def read(self, dosages: npt.NDArray[np.float64]) -> None:
        if dosages.size == 0:
            return
        read_dosages(
            str(self.file_path), dosages, self.sample_indices, self.variant_indices
        )

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass
