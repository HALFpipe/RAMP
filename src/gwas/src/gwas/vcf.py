# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from numpy import typing as npt

from .log import logger
from .z import CompressedTextReader


@dataclass
class Variant:
    position: int
    reference_allele: str
    alternative_allele: str


class VCFFile(CompressedTextReader):
    mandatory_columns = [
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
    ]

    def __init__(self, file_path: Path | str, samples: list[str] | None = None) -> None:
        super().__init__(file_path)

        logger.info(f'Scanning "{str(file_path)}"')

        # read header information and example line
        header: str | None = None
        line: str | None = None
        self.variant_count = 0
        with self as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    if not line.startswith("##"):
                        header = line
                    continue
                self.variant_count += 1

        if not isinstance(header, str):
            raise ValueError

        if not isinstance(line, str):
            raise ValueError

        # extract and check column names
        columns = header.strip().removeprefix("#").split()
        if columns[: len(self.mandatory_columns)] != self.mandatory_columns:
            raise ValueError

        # set properties
        self.samples: list[str] = columns[len(self.mandatory_columns) :]
        self.sample_indices: list[int] | None = None

        if samples is not None:
            self.sample_indices = [self.samples.index(sample) for sample in samples]
            self.samples = samples

        self.sample_count = len(self.samples)

        self.chromosome_column_index = columns.index("CHROM")
        self.position_column_index = columns.index("POS")
        self.reference_allele_column_index = columns.index("REF")
        self.alternative_allele_column_index = columns.index("ALT")
        self.info_column_index = columns.index("INFO")
        self.format_column_index = columns.index("FORMAT")

        examples = line.split()

        self.chromosome: int | str = examples[self.chromosome_column_index]
        if self.chromosome.isdigit():
            self.chromosome = int(self.chromosome)

        example_format = examples[self.format_column_index]
        fields = example_format.split(":")
        self.field_count = len(fields)
        self.dosage_field_index = fields.index("DS")

    def read(
        self,
        dosages: npt.NDArray,
        only_snps: bool = False,
        predicate: Callable[[npt.NDArray], bool] | None = None,
    ) -> list[Variant]:
        if self.file_handle is None:
            raise ValueError

        if dosages.size == 0:
            return list()  # nothing to do

        if dosages.shape[1] != self.sample_count:
            raise ValueError

        variants: list[Variant] = list()

        n_mandatory_columns = len(self.mandatory_columns)

        variant_index: int = 0

        for line in self.file_handle:
            if line.startswith("#"):
                continue

            tokens = line.split(maxsplit=n_mandatory_columns)

            # parse metadata
            reference_allele = tokens[self.reference_allele_column_index]
            alternative_allele = tokens[self.alternative_allele_column_index]
            if only_snps:
                if len(reference_allele) != 1 or len(alternative_allele) != 1:
                    continue  # skip line

            # parse dosages
            fields = tokens[-1].replace(":", "\t").split()
            dosage_fields = fields[self.dosage_field_index :: self.field_count]
            if self.sample_indices is not None:
                dosage_fields = [dosage_fields[i] for i in self.sample_indices]
            dosages[variant_index, :] = dosage_fields

            if (
                predicate is not None
                and predicate(dosages[variant_index, :]) is not True
            ):
                continue  # skip line

            variants.append(
                Variant(
                    position=int(tokens[self.position_column_index]),
                    reference_allele=reference_allele,
                    alternative_allele=alternative_allele,
                )
            )
            variant_index += 1

            if variant_index >= dosages.shape[0]:
                break

        return variants
