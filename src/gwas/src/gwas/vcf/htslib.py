from pathlib import Path

import numpy as np
import pandas as pd
from numpy import typing as npt

from gwas.vcf._htslib import read, read_vcf_records  

from .base import VCFFile

variant_columns = [
    "chromosome_int",
    "position",
    "reference_allele",
    "alternate_allele",
    "is_imputed",
    "alternate_allele_frequency",
    "minor_allele_frequency",
    "r_squared",
]


class HTSLIBVCFFile(VCFFile):
    def __init__(self, file_path: str | Path) -> None:
        super().__init__(file_path)
        self.file_path = Path(file_path)
        self.vcf_variants: pd.DataFrame
        self.variant_indices: npt.NDArray[np.uint32]
        self.create_dataframe()

    def create_dataframe(self):
        variants, samples = read_vcf_records(str(self.file_path))
        self.samples = samples
        self.sample_indices = np.array(
            [i for i, s in enumerate(self.samples)],
            dtype=np.uint32,
        )
        self.vcf_variants = pd.DataFrame(variants, columns=variant_columns)

        self.vcf_variants["chromosome_int"] = self.vcf_variants["chromosome_int"].astype(
            "int64"
        ).astype('category')
        self.vcf_variants["reference_allele"] = self.vcf_variants[
            "reference_allele"
        ].astype("category")
        self.vcf_variants["alternate_allele"] = self.vcf_variants[
            "alternate_allele"
        ].astype("category")
        self.vcf_variants["position"] = self.vcf_variants["position"].astype("uint32")

        self.variant_indices = np.arange(self.vcf_variant_count, dtype=np.uint32)

    def read(self, dosages: npt.NDArray[np.float64]) -> None:
        if dosages.size == 0:
            return
        read(str(self.file_path), dosages, self.sample_indices, self.variant_indices)
