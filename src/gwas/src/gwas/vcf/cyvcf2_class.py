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


class CyVCF2VCFFile2(VCFFile):
    def __init__(self, file_path: str | Path, samples: set[str] | None = None) -> None:
        super().__init__()
        # if isinstance(self.file_path, Path):
        #     file_path = str(file_path)
        self.file_path = str(file_path)
        self.samples = list(samples) if samples else []
        self.vcf: VCF = None
        self.vcf_variants = None
        self.variant_indices = np.array([], dtype=np.uint32)
        self.sample_indices: npt.NDArray[np.uint32] = np.array([], dtype=np.uint32)
        self.initialized = False

    def initialize_vcf_file(self):
        if not self.initialized or self.vcf is None:
            if self.file_path.endswith(".zst"):
                with CompressedBytesReader(file_path=self.file_path) as f:
                    self.vcf = VCF(f)
            else:
                self.vcf = VCF(self.file_path)

            if self.samples:
                self.vcf.set_samples(self.samples)
            self.vcf_samples = list(self.vcf.samples)
            print("Samples after setting:", self.vcf_samples)
            self.sample_indices = np.array(
                [
                    self.vcf_samples.index(s)
                    for s in self.samples
                    if s in self.vcf_samples
                ],
                dtype=np.uint32,
            )
            print("Sample indices:", self.sample_indices)
            self.initialized = True

        # self.vcf_samples = self.vcf.samples

        # self.samples: list[str] = (
        #     list(samples) if samples else []
        # )  # needed for sample count

        # self.sample_indices = np.array([], dtype=np.uint32)
        # self.sample_indices = np.array(
        #     [self.vcf_samples.index(s) for s in self.samples if s in self.vcf_samples],
        #     dtype=np.uint32,
        # )
        # self.variant_indices = np.array([], dtype=np.uint32)

    def create_dataframe(self) -> pd.DataFrame:
        # Convert VCF data from cyvcf2 to a DataFrame
        # self.vcf.set_samples(self.samples)
        if not self.vcf:
            self.initialize_vcf_file()
            print("DATAFRAME CREATION")
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
            self.vcf_variants = pd.DataFrame(variants, columns=variant_columns)
            self.close()

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
        # if dosages.shape != (len(self.variant_indices), len(self.sample_indices)):
        #     raise ValueError(
        #         f"""Expected dosages array shape {(
        #         len(self.variant_indices),
        #         len(self.sample_indices))}, but got {dosages.shape}"""
        #     )
        if dosages.size == 0:
            return
        self.initialize_vcf_file()

        variant_count = 0
        for variant in self.vcf:
            if variant_count in self.variant_indices:
                for j, sample_idx in enumerate(self.sample_indices):
                    try:
                        dosages[variant_count, j] = (
                            variant.format("DS")[sample_idx][0]
                            if "DS" in variant.FORMAT
                            else np.nan
                        )
                    except IndexError:
                        dosages[variant_count, j] = np.nan
                variant_count += 1
                if variant_count > max(
                    self.variant_indices
                ):  # müssen max benutzen und nicht len!
                    break

        self.close()

        # for i, variant_idx in enumerate(self.variant_indices):
        #    variant = next((
        # v for j, v in enumerate(self.vcf) if j == variant_idx), None
        # )
        #    if variant is not None:
        #        dosages[i, :] = [
        #            float(variant.format("DS")[idx][0]) for idx in self.sample_indices
        #        ]

    def close(self):
        if self.vcf is not None:
            self.vcf.close()
            self.vcf = None
            self.initialized = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return super().__exit__(exc_type, exc_value, traceback)


class CyVCF2VCFFile(VCFFile):
    def __init__(
        self,
        file_path: str | Path,
        samples: set[str] | None,
    ) -> None:
        super().__init__()
        self.file_path = str(file_path)
        self.vcf: VCF = None
        self.vcf_variants = None
        self.vcf_variants: pd.DataFrame
        self.variant_indices: npt.NDArray[np.uint32]

        self.create_dataframe()
        # we set self.samples after the creation of our self.vcf object
        # in make dataframe function to obtain all samples if samples
        # is not specified we get all samples
        self.samples = list(samples) if samples else self.vcf.samples
        self.sample_indices = np.array(
            [i for i, s in enumerate(self.vcf.samples) if s in self.samples],
            dtype=np.uint32,
        )

    def return_vcf_object(self):
        if self.file_path.endswith(".zst"):
            # with CompressedBytesReader(file_path=self.file_path) as f:
            # vcf = VCF(f)
            return None
        else:
            vcf = VCF(self.file_path)  # for .gz case or raw vcf

        return vcf

    def create_dataframe(self) -> pd.DataFrame:
        # Convert VCF data from cyvcf2 to a DataFrame

        # vcf_df = self.return_vcf_object()
        self.vcf = self.return_vcf_object()
        # we do not need to set samples here since we
        # do not use sample columns to make the dataframe
        # if self.samples:
        # vcf_df.set_samples(self.samples)
        # self.vcf.set_samples(self.samples)

        print("DATAFRAME CREATION")
        variants = []
        # for _, variant in enumerate(vcf_df):
        for _, variant in enumerate(self.vcf):
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
        self.vcf_variants = pd.DataFrame(variants, columns=variant_columns)
        self.variant_indices = np.arange(self.vcf_variant_count, dtype=np.uint32)

    def read(self, dosages: npt.NDArray) -> None:
        if dosages.size == 0:
            return
        if dosages.shape[1] != self.sample_count:
            raise ValueError(
                "The output array does not match the number of samples "
                f"({dosages.shape[1]} != {self.sample_count})"
            )
        vcf_read = self.return_vcf_object()
        if self.samples:
            vcf_read.set_samples(self.samples)

        variant_count = 0
        for variant in vcf_read:
            if variant_count in self.variant_indices:
                if "DS" in variant.FORMAT:
                    dosage_fields = variant.format("DS")
                    dosage_fields = self.process_dosage_fields(
                        dosage_fields=dosage_fields, sample_indices=self.sample_indices
                    )
                    if dosage_fields.shape[0] != dosages.shape[1]:
                        raise ValueError(
                            f"""Shape of dosage_fields does not match
                            the number of samples"""
                            f"({dosage_fields.shape[0]} != {dosages.shape[1]})"
                        )
                    dosages[variant_count, :] = dosage_fields
                else:
                    dosages[variant_count, :] = np.nan
            # update variant count
            variant_count += 1
            if variant_count > max(self.variant_indices):
                break

    def process_dosage_fields(self, dosage_fields, sample_indices) -> np.ndarray:
        if sample_indices is not None:
            dosage_fields = np.array([dosage_fields[i] for i in sample_indices])
        else:
            dosage_fields = np.array(dosage_fields)
        if dosage_fields.ndim > 1:
            dosage_fields = dosage_fields.flatten()
        return dosage_fields

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return super().__exit__(exc_type, exc_value, traceback)
