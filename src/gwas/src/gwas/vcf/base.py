# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import ClassVar, NamedTuple, Self

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm import tqdm

from ..compression.pipe import CompressedTextReader, load_from_cache, save_to_cache
from ..log import logger
from ..utils import Pool, chromosome_to_int


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

        if is_imputed:
            r_squared = float(info.get("R2", np.nan))
        else:
            r_squared = float(info.get("ER2", np.nan))

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


class VCFFile(CompressedTextReader):
    mandatory_columns: ClassVar[tuple[str, ...]] = (
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
    )

    chromosome_column_index: ClassVar[int] = mandatory_columns.index("CHROM")
    position_column_index: ClassVar[int] = mandatory_columns.index("POS")
    reference_allele_column_index: ClassVar[int] = mandatory_columns.index("REF")
    alternate_allele_column_index: ClassVar[int] = mandatory_columns.index("ALT")
    info_column_index: ClassVar[int] = mandatory_columns.index("INFO")
    format_column_index: ClassVar[int] = mandatory_columns.index("FORMAT")

    metadata_column_indices: ClassVar[npt.NDArray[np.uint32]] = np.array(
        [
            chromosome_column_index,
            position_column_index,
            reference_allele_column_index,
            alternate_allele_column_index,
            info_column_index,
            format_column_index,
        ],
        dtype=np.uint32,
    )

    chromosome: int | str

    vcf_samples: list[str]
    samples: list[str]
    sample_indices: npt.NDArray[np.uint32]

    vcf_variants: pd.DataFrame
    variant_indices: npt.NDArray[np.uint32]
    allele_frequency_columns: list[str]

    minor_allele_frequency_cutoff: float
    r_squared_cutoff: float

    def __init__(self, file_path: Path | str) -> None:
        super().__init__(file_path)

        self.allele_frequency_columns = [
            "minor_allele_frequency",
            "alternate_allele_frequency",
        ]

        self.minor_allele_frequency_cutoff = -np.inf
        self.r_squared_cutoff = -np.inf

        # Read header information and example line.
        self.header_length: int = 0
        column_names_line: str | None = None
        example_lines: list[str] = list()
        with self as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    self.header_length += len(line)
                    if not line.startswith("##"):
                        column_names_line = line
                    continue
                if len(example_lines) < 2:
                    example_lines.append(line)
                    continue
                break

        if not isinstance(column_names_line, str):
            raise ValueError(f"Could not find column names in {file_path}")

        if not len(example_lines) > 0:
            raise ValueError(f"Could not find example lines in {file_path}")

        self.example_lines = example_lines

        # Extract and check column names.
        self.columns = column_names_line.strip().removeprefix("#").split()
        if tuple(self.columns[: len(self.mandatory_columns)]) != self.mandatory_columns:
            raise ValueError

        self.metadata_column_count = len(self.mandatory_columns)
        self.vcf_samples: list[str] = self.columns[len(self.mandatory_columns) :]
        self.set_samples(set(self.vcf_samples))

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def vcf_variant_count(self) -> int:
        return len(self.vcf_variants.index)

    @property
    def variant_count(self) -> int:
        return self.variant_indices.size

    @property
    def variants(self) -> pd.DataFrame:
        return self.vcf_variants.iloc[self.variant_indices, :]

    def reset_variants(self) -> None:
        self.set_variants_from_cutoffs()

    def set_variants_from_cutoffs(
        self,
        minor_allele_frequency_cutoff: float = -np.inf,
        r_squared_cutoff: float = -np.inf,
    ):
        def greater_or_close(series: pd.Series, cutoff: float) -> npt.NDArray[np.bool_]:
            value = np.asanyarray(series.values)
            return (value >= cutoff) | np.isclose(value, cutoff) | np.isnan(value)

        allele_frequency_frame = self.vcf_variants[self.allele_frequency_columns].copy()
        is_major = allele_frequency_frame.values > 0.5
        allele_frequency_frame.values[is_major] = (
            1 - allele_frequency_frame.values[is_major]
        )
        allele_frequency = allele_frequency_frame.max(axis="columns")
        self.variant_indices = np.flatnonzero(
            greater_or_close(
                allele_frequency,
                minor_allele_frequency_cutoff,
            )
            & greater_or_close(self.vcf_variants.r_squared, r_squared_cutoff)
        ).astype(np.uint32)

        logger.debug(
            f"After filtering with "
            f'"minor_allele_frequency >= {minor_allele_frequency_cutoff}" and '
            f'"r_squared >= {r_squared_cutoff}" '
            f"{self.variant_indices.size} of {self.vcf_variant_count} variants remain."
        )
        self.minor_allele_frequency_cutoff = minor_allele_frequency_cutoff
        self.r_squared_cutoff = r_squared_cutoff

    def set_samples(self, samples: set[str]) -> None:
        self.sample_indices = np.fromiter(
            (i for i, sample in enumerate(self.vcf_samples) if sample in samples),
            dtype=np.uint32,
        )
        self.samples = [sample for sample in self.vcf_samples if sample in samples]

    def save_to_cache(self, cache_path: Path) -> None:
        save_to_cache(cache_path, self.cache_key(self.file_path), self)

    @abstractmethod
    def read(
        self,
        dosages: npt.NDArray,
    ) -> None: ...

    @staticmethod
    def cache_key(vcf_path: Path) -> str:
        stem = vcf_path.name.split(".")[0]
        cache_key = f"{stem}.vcf-metadata"
        return cache_key

    @classmethod
    def load_from_cache(cls, cache_path: Path, vcf_path: Path) -> VCFFile:
        return load_from_cache(cache_path, cls.cache_key(vcf_path))

    @staticmethod
    def from_path(
        file_path: Path | str,
        samples: set[str] | None = None,
        engine: str = "cpp",
    ) -> VCFFile:
        if engine == "python":
            from .python import PyVCFFile

            vcf_file: VCFFile = PyVCFFile(file_path)
        elif engine == "cpp":
            from .cpp import CppVCFFile

            vcf_file = CppVCFFile(file_path)
        else:
            raise ValueError

        if samples is not None:
            vcf_file.set_samples(samples)

        return vcf_file

    @staticmethod
    def make_data_frame(vcf_variants: list[Variant]) -> pd.DataFrame:
        data_frame = pd.DataFrame(vcf_variants, columns=variant_columns)

        data_frame["position"] = data_frame["position"].astype(np.uint32)

        for column in [
            "chromosome_int",
            "reference_allele",
            "alternate_allele",
            "format_str",
        ]:
            data_frame[column] = data_frame[column].astype("category")

        return data_frame


def load_vcf(
    cache_path: Path,
    vcf_path: Path,
) -> VCFFile:
    vcf_file: VCFFile | None = VCFFile.load_from_cache(cache_path, vcf_path)
    if vcf_file is None:
        vcf_file = VCFFile.from_path(vcf_path)
        vcf_file.save_to_cache(cache_path)
    else:
        logger.debug(f'Cached VCF file metadata for "{VCFFile.cache_key(vcf_path)}"')
    return vcf_file


def calc_vcf(
    vcf_paths: list[Path],
    cache_path: Path,
) -> list[VCFFile]:
    processes = cpu_count() // 3
    processes = min(processes, len(vcf_paths))
    with Pool(processes=processes, maxtasksperchild=1) as pool:
        vcf_files = list(
            tqdm(
                pool.imap(partial(load_vcf, cache_path), vcf_paths),
                total=len(vcf_paths),
                unit="files",
                desc="loading vcf metadata",
            )
        )
    return vcf_files
