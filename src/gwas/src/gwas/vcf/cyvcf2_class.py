# -*- coding: utf-8 -*-
from pathlib import Path
from types import TracebackType

import numpy as np
import pandas as pd
from cyvcf2 import VCF
from numpy import typing as npt

from ..utils import chromosome_to_int, parse_chromosome
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
            return None
        else:
            vcf = VCF(self.file_path)  # for .gz case or raw vcf

        return vcf

    def create_dataframe(self) -> pd.DataFrame:
        # Convert VCF data from cyvcf2 to a DataFrame
        self.vcf = self.return_vcf_object()

        print("DATAFRAME CREATION")
        variants = []
        self.variant_dosage_fields = []
        for _, variant in enumerate(self.vcf):
            r2_value = variant.INFO.get(
                "ER2", variant.INFO.get("R2", np.nan)
            )  # use ER2 value if present
            variants.append(
                [
                    chromosome_to_int(parse_chromosome(variant.CHROM)),
                    variant.POS,
                    variant.REF,
                    variant.ALT[0] if variant.ALT else "",
                    variant.INFO.get("IMPUTED", False),
                    variant.INFO.get("AF", np.nan),
                    variant.INFO.get("MAF", np.nan),
                    # variant.INFO.get("R2", np.nan),
                    r2_value,
                    (":").join(variant.FORMAT),
                ]
            )
            if "DS" in variant.FORMAT:
                self.variant_dosage_fields.append(variant.format("DS"))
            else:
                self.variant_dosage_fields.append(np.nan)
        self.vcf_variants = pd.DataFrame(variants, columns=variant_columns)

        self.vcf_variants["chromosome_int"] = self.vcf_variants["chromosome_int"].astype(
            "category"
        )
        self.vcf_variants["reference_allele"] = self.vcf_variants[
            "reference_allele"
        ].astype("category")
        self.vcf_variants["alternate_allele"] = self.vcf_variants[
            "alternate_allele"
        ].astype("category")
        self.vcf_variants["format_str"] = self.vcf_variants["format_str"].astype(
            "category"
        )
        self.vcf_variants["position"] = self.vcf_variants["position"].astype("uint32")

        self.variant_indices = np.arange(self.vcf_variant_count, dtype=np.uint32)

    def read(self, dosages: npt.NDArray) -> None:
        if dosages.size == 0:
            return
        if dosages.shape[1] != self.sample_count:
            raise ValueError(
                "The output array does not match the number of samples "
                f"({dosages.shape[1]} != {self.sample_count})"
            )
        # vcf_read = self.return_vcf_object()
        # if self.samples:
        #    vcf_read.set_samples(self.samples)

        # variant_index_iter = iter(self.variant_indices)
        # current_index = self.get_next_variant_index(variant_index_iter)

        pos_in_dosage = 0
        for idx in self.variant_indices:
            dosage_fields = self.variant_dosage_fields[idx]
            if dosage_fields is not np.nan:
                processed_dosage_fields = self.process_dosage_fields(
                    dosage_fields, self.sample_indices
                )
                if processed_dosage_fields[0] != dosages.shape[1]:
                    raise ValueError(
                        f"Shape of dosage_fields does not match the number of samples "
                        f"({dosage_fields.shape[0]} != {dosages.shape[1]})"
                    )
                dosages[pos_in_dosage, :] = processed_dosage_fields
            else:
                dosages[pos_in_dosage, :] = np.nan
            pos_in_dosage += 1
        # for variant_count, variant in enumerate(vcf_read):
        #     if variant_count == current_index:
        #         self.process_variant(variant, pos_in_dosage, dosages)
        #         pos_in_dosage += 1
        #         current_index = self.get_next_variant_index(variant_index_iter)
        #         if current_index is None:
        #             break

    def process_variant(self, variant, pos_in_dosage, dosages) -> None:
        if "DS" in variant.FORMAT:
            dosage_fields = variant.format("DS")
            dosage_fields = self.process_dosage_fields(
                dosage_fields, self.sample_indices
            )
            if dosage_fields.shape[0] != dosages.shape[1]:
                raise ValueError(
                    f"Shape of dosage_fields does not match the number of samples "
                    f"({dosage_fields.shape[0]} != {dosages.shape[1]})"
                )
            dosages[pos_in_dosage, :] = dosage_fields
        else:
            dosages[pos_in_dosage, :] = np.nan

    def get_next_variant_index(self, variant_index_iter):
        try:
            return next(variant_index_iter)
        except StopIteration:
            return None

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
