from abc import abstractmethod
from enum import Enum, auto
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm.auto import tqdm

from ..compression.pipe import CompressedTextReader, load_from_cache, save_to_cache
from ..log import logger
from ..utils import (
    IterationOrder,
    chromosome_to_int,
    make_pool_or_null_context,
    make_variant_mask,
)
from .variant import Variant


class Engine(Enum):
    python = auto()
    cpp = auto()


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


def read_header(reader: CompressedTextReader) -> tuple[int, list[str], list[str]]:
    header_length: int = 0
    column_names_line: str | None = None
    example_lines: list[str] = list()
    with reader as file_handle:
        for line in file_handle:
            if line.startswith("#"):
                header_length += len(line)
                if not line.startswith("##"):
                    column_names_line = line
                continue
            if len(example_lines) < 2:
                example_lines.append(line)
                continue
            break

    if not isinstance(column_names_line, str):
        raise ValueError(f"Could not find column names in {reader.file_path}")
    if not len(example_lines) > 0:
        raise ValueError(f"Could not find example lines in {reader.file_path}")

    columns = column_names_line.strip().removeprefix("#").split()
    return header_length, columns, example_lines


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
        self.header_length, self.columns, self.example_lines = read_header(self)

        # Extract and check column names.
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
    def variant_mask(self) -> npt.NDArray[np.bool_]:
        variant_mask = np.zeros(self.vcf_variant_count, dtype=np.bool_)
        variant_mask[self.variant_indices] = True
        return variant_mask

    @property
    def variants(self) -> pd.DataFrame:
        return self.vcf_variants.iloc[self.variant_indices, :]

    def set_variants_from_cutoffs(
        self,
        minor_allele_frequency_cutoff: float = -np.inf,
        r_squared_cutoff: float = -np.inf,
        aggregate_func: str = "max",
    ) -> None:
        variant_mask = make_variant_mask(
            self.vcf_variants[self.allele_frequency_columns],
            self.vcf_variants.r_squared,
            minor_allele_frequency_cutoff,
            r_squared_cutoff,
            aggregate_func=aggregate_func,
        )
        self.variant_indices = np.flatnonzero(variant_mask).astype(np.uint32)

        logger.debug(
            f"After filtering with "
            f'"minor_allele_frequency >= {minor_allele_frequency_cutoff}" and '
            f'"r_squared >= {r_squared_cutoff}" '
            f"{self.variant_indices.size} of {self.vcf_variant_count} variants remain",
            stack_info=True,
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
        dosages: npt.NDArray[np.float64],
    ) -> None: ...

    @staticmethod
    def cache_key(vcf_path: Path) -> str:
        stem = vcf_path.name.split(".")[0]
        cache_key = f"{stem}.vcf-metadata"
        return cache_key

    @classmethod
    def load_from_cache(cls, cache_path: Path, vcf_path: Path) -> "VCFFile":
        v = load_from_cache(cache_path, cls.cache_key(vcf_path))
        if not isinstance(v, VCFFile):
            raise ValueError(f"Expected VCFFile, got {type(v)}: {v}")
        v.file_path = vcf_path
        return v

    @staticmethod
    def from_path(
        file_path: Path | str,
        samples: set[str] | None = None,
        engine: Engine = Engine.cpp,
    ) -> "VCFFile":
        if engine == Engine.python:
            from .python import PyVCFFile

            vcf_file: VCFFile = PyVCFFile(file_path)
        elif engine == Engine.cpp:
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
    engine: Engine = Engine.cpp,
) -> VCFFile:
    vcf_file: VCFFile | None = None
    try:
        vcf_file = VCFFile.load_from_cache(cache_path, vcf_path)
    except ValueError:
        pass
    if vcf_file is None:
        vcf_file = VCFFile.from_path(vcf_path, engine=engine)
        vcf_file.save_to_cache(cache_path)
    else:
        logger.debug(f'Cached VCF file metadata for "{VCFFile.cache_key(vcf_path)}"')
    return vcf_file


def calc_vcf(
    vcf_paths: list[Path],
    cache_path: Path,
    num_threads: int = 1,
    engine: Engine = Engine.cpp,
) -> list[VCFFile]:
    pool, iterator = make_pool_or_null_context(
        vcf_paths,
        partial(load_vcf, cache_path, engine=engine),
        num_threads=num_threads,
        iteration_order=IterationOrder.UNORDERED,
    )
    with pool:
        vcf_files = list(
            tqdm(
                iterator,
                total=len(vcf_paths),
                unit="files",
                desc="loading vcf metadata",
            )
        )
    vcf_files.sort(key=lambda v: chromosome_to_int(v.chromosome))
    return vcf_files
