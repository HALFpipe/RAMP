# -*- coding: utf-8 -*-
from __future__ import annotations
from contextlib import AbstractContextManager
import gzip
from pathlib import Path
from typing import TextIO

from numpy import typing as npt


class VCFFile:
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

    def __init__(
        self, file_path: Path | str, samples: list[str] | None = None
    ) -> None:
        self.file_path = file_path

        # read header information and example line
        header: str | None = None
        line: str | None = None
        self.variant_count = 0
        with gzip.open(self.file_path, "rt") as file_handle:
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
        if columns[:len(self.mandatory_columns)] != self.mandatory_columns:
            raise ValueError

        # set properties
        self.samples: list[str] = columns[len(self.mandatory_columns):]
        self.sample_indices: list[int] | None = None

        if samples is not None:
            self.sample_indices = [
                self.samples.index(sample) for sample in samples
            ]
            self.samples = samples

        self.sample_count = len(self.samples)

        self.chromosome_column_index = columns.index("CHROM")
        self.position_column_index = columns.index("POS")
        self.reference_allele_column_index = columns.index("REF")
        self.alternative_allele_column_index = columns.index("ALT")

        self.format_column_index = columns.index("FORMAT")

        examples = line.split()

        self.chromosome: int | str = examples[self.chromosome_column_index]
        if self.chromosome.isdigit():
            self.chromosome = int(self.chromosome)

        example_format = examples[self.format_column_index]
        fields = example_format.split(":")
        self.field_count = len(fields)
        self.dosage_field_index = fields.index("DS")


class GenotypeReader(AbstractContextManager):
    def __init__(self, vcf_file: VCFFile) -> None:
        self.vcf_file = vcf_file

        self.file_handle: TextIO | None = None

    def __enter__(self) -> GenotypeReader:
        self.file_handle = gzip.open(self.vcf_file.file_path, "rt")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if isinstance(self.file_handle, TextIO):
            self.file_handle.close()
            self.file_handle = None

    def read(self, dosages: npt.NDArray):
        if not isinstance(self.file_handle, TextIO):
            raise ValueError

        if dosages.size == 0:
            return  # nothing to do

        if dosages.shape[0] != self.vcf_file.sample_count:
            raise ValueError

        positions: list[int] = list()
        alleles: list[tuple[str, str]] = list()

        n_mandatory_columns = len(self.vcf_file.mandatory_columns)

        variant_index = 0

        for line in self.file_handle:
            if line.startswith("#"):
                continue

            tokens = line.split(maxsplit=n_mandatory_columns)

            positions[variant_index] = int(
                tokens[self.vcf_file.position_column_index]
            )

            reference_allele = tokens[
                self.vcf_file.reference_allele_column_index
            ]
            alternative_allele = tokens[
                self.vcf_file.alternative_allele_column_index
            ]
            alleles.append(
                (reference_allele, alternative_allele)
            )

            fields = tokens[-1].replace(":", "\t").split()
            dosage_fields = fields[
                self.vcf_file.dosage_field_index::self.vcf_file.field_count
            ]
            if self.vcf_file.sample_indices is not None:
                dosage_fields = [
                    dosage_fields[i] for i in self.vcf_file.sample_indices
                ]
            dosages[variant_index, :] = dosage_fields

            variant_index += 1
            if variant_index >= dosages.shape[0]:
                break

        return variant_index
