# -*- coding: utf-8 -*-
from pathlib import Path
from types import TracebackType

import numpy as np
import pandas as pd
from cyvcf2 import VCF
from numpy import typing as npt

from gwas.compression.pipe import CompressedBytesReader

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
    "format_str",
]


class CyVCF2VCFFile(VCFFile):
    def __init__(self, file_path: str | Path, samples: set[str] | None = None) -> None:
        super().__init__()
        self.path = file_path
        if isinstance(file_path, Path):
            file_path = str(file_path)
        if file_path.endswith(".zst"):
            with CompressedBytesReader(file_path=file_path) as f:
                self.vcf = VCF(f)
        else:
            self.vcf = VCF(file_path)

        # if samples is not None:
        #     # sample_list = ",".join(samples).encode("utf-8")
        #     sample_list = list(samples)
        #     # self.vcf.set_samples(sample_list)
        #     #self.vcf.set_samples(",".join(samples).encode("utf-8"))
        #     self.vcf.set_samples(sample_list)

        self.vcf_samples = self.vcf.samples  # renamed to samples
        print(len(self.vcf_samples))

        # self.all_variants = [v for v in self.vcf]
        # self.vcf_variants = self.make_data_frame(self.vcf)  # all variants in the file

        self.samples: list[str] = (
            list(samples) if samples else []
        )  # needed for sample count

        # self.sample_indices = np.array([], dtype=np.uint32)
        self.sample_indices = np.array(
            [self.vcf_samples.index(s) for s in self.samples if s in self.vcf_samples],
            dtype=np.uint32,
        )
        self.variant_indices = np.array([], dtype=np.uint32)

        self.vcf_variants = self.create_dataframe()

    # def set_samples(self, samples: set[str]):
    #    super().set_samples(samples)
    #    self.samples = [sample for sample in self.samples if sample in samples]
    #    self.sample_indices = np.array(
    #        [self.vcf_samples.index(s) for s in self.samples if s in self.vcf_samples],
    #        dtype=np.uint32,
    #    )

    def create_dataframe(self) -> pd.DataFrame:
        # Convert VCF data from cyvcf2 to a DataFrame
        # self.vcf.set_samples(self.samples)
        variants = []
        for variant in self.vcf:
            variants.append(
                [
                    variant.CHROM,
                    variant.POS,
                    variant.REF,
                    variant.ALT[0] if variant.ALT else "",
                    variant.INFO.get("IMPUTED", False),
                    variant.INFO.get("AF", float("nan")),
                    variant.INFO.get("MAF", float("nan")),
                    variant.INFO.get("RSQ", float("nan")),
                    variant.FORMAT,
                ]
            )
        return pd.DataFrame(variants, columns=variant_columns)

    def set_variants_from_cutoffs(
        self,
        minor_allele_frequency_cutoff: float = -np.inf,
        r_squared_cutoff: float = -np.inf,
    ):
        cutoff_indices = [
            i
            for i, v in enumerate(self.vcf)
            if float(v.INFO.get("MAF")) > minor_allele_frequency_cutoff
            and float(v.INFO.get("R2")) > r_squared_cutoff
        ]
        self.variant_indices = np.array(cutoff_indices, dtype=np.uint32)

    # def read(self, dosages: npt.NDArray) -> None:
    #    for i, variant_idx in enumerate(self.variant_indices):
    #        variant = self.all_variants[variant_idx]
    #        dosages[i, :] = [
    #            variant.format("DS")[idx][0] for idx in self.sample_indices
    #        ]  # for loop rausnehmen?
    def read(self, dosages: npt.NDArray) -> None:
        # if dosages.shape != (self.variant_count, self.sample_count):
        #     raise ValueError(
        #         f"Expected shape {(self.variant_count, self.sample_count)} for "
        #         f"variable `dosages` but received {dosages.shape}"
        #     )
        if isinstance(self.path, Path):
            file_path = str(self.path)
        if file_path.endswith(".zst"):
            with CompressedBytesReader(file_path=file_path) as f:
                self.vcf = VCF(f)
        else:
            self.vcf = VCF(self.path)

        for i, variant_idx in enumerate(self.variant_indices):
            variant = next((v for j, v in enumerate(self.vcf) if j == variant_idx), None)
            if variant is not None:
                dosages[i, :] = [
                    float(variant.format("DS")[idx][0]) for idx in self.sample_indices
                ]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return super().__exit__(exc_type, exc_value, traceback)
