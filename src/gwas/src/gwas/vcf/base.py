from abc import abstractmethod
from contextlib import AbstractContextManager
from enum import Enum, auto
from functools import partial
from typing import ClassVar

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm.auto import tqdm
from upath import UPath

from ..compression.cache import load_from_cache, save_to_cache
from ..compression.pipe import CompressedTextReader
from ..log import logger
from ..mem.data_frame import SharedDataFrame
from ..mem.wkspace import SharedWorkspace
from ..utils.genetics import chromosome_to_int, make_variant_mask
from ..utils.multiprocessing import IterationOrder, make_pool_or_null_context
from .variant import Variant


class Engine(Enum):
    python = auto()
    cpp = auto()
    htslib = auto()


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


class VCFFile(AbstractContextManager):
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

    file_path: UPath

    chromosome: int | str

    vcf_samples: list[str]
    samples: list[str]
    sample_indices: npt.NDArray[np.uint32]

    shared_vcf_variants: SharedDataFrame
    variant_indices: npt.NDArray[np.uint32]
    allele_frequency_columns: list[str]

    minor_allele_frequency_cutoff: float
    r_squared_cutoff: float

    def __init__(self, file_path: UPath) -> None:
        self.file_path = file_path

        self.allele_frequency_columns = [
            "minor_allele_frequency",
            "alternate_allele_frequency",
        ]

        self.minor_allele_frequency_cutoff = -np.inf
        self.r_squared_cutoff = -np.inf

    @property
    def vcf_variants(self) -> pd.DataFrame:
        return self.shared_vcf_variants.to_pandas()

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def vcf_variant_count(self) -> int:
        return self.shared_vcf_variants.shape[0]

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

    def save_to_cache(self, cache_path: UPath, num_threads: int) -> None:
        save_to_cache(cache_path, self.cache_key(self.file_path), self, num_threads)

    def clear_allele_frequency_columns(self) -> None:
        allele_frequency_columns = self.allele_frequency_columns

        # Keep the alternate_allele_frequency column
        if "alternate_allele_frequency" in allele_frequency_columns:
            allele_frequency_columns.remove("alternate_allele_frequency")

        for column in self.shared_vcf_variants.columns:
            if column.name in self.allele_frequency_columns:
                column.free()
        self.shared_vcf_variants.columns = [
            column
            for column in self.shared_vcf_variants.columns
            if column.name not in self.allele_frequency_columns
        ]

    def free(self) -> None:
        self.shared_vcf_variants.free()

    @abstractmethod
    def read(
        self,
        dosages: npt.NDArray[np.float64],
    ) -> None: ...

    @staticmethod
    def cache_key(vcf_path: UPath) -> str:
        stem = vcf_path.name.split(".")[0]
        cache_key = f"{stem}.vcf-metadata"
        return cache_key

    @classmethod
    def load_from_cache(
        cls, cache_path: UPath, vcf_path: UPath, sw: SharedWorkspace
    ) -> "VCFFile":
        v = load_from_cache(cache_path, cls.cache_key(vcf_path), sw)
        if not isinstance(v, VCFFile):
            raise ValueError(f"Expected VCFFile, got {type(v)}: {v}")
        v.file_path = vcf_path
        return v

    @staticmethod
    def from_path(
        file_path: UPath,
        sw: SharedWorkspace,
        samples: set[str] | None = None,
        engine: Engine = Engine.cpp,
    ) -> "VCFFile":
        if engine == Engine.python:
            from .python import PyVCFFile

            vcf_file: VCFFile = PyVCFFile(file_path, sw)
        elif engine == Engine.cpp:
            from .cpp import CppVCFFile

            vcf_file = CppVCFFile(file_path, sw)
        elif engine == Engine.htslib:
            from .htslib import HtslibVCFFile

            vcf_file = HtslibVCFFile(file_path, sw)
        else:
            raise ValueError

        if samples is not None:
            vcf_file.set_samples(samples)

        return vcf_file

    @classmethod
    def make_shared_data_frame(
        cls, vcf_variants: list[Variant], sw: SharedWorkspace
    ) -> SharedDataFrame:
        return SharedDataFrame.from_pandas(cls.make_data_frame(vcf_variants), sw)

    @staticmethod
    def make_data_frame(vcf_variants: list[Variant]) -> pd.DataFrame:
        data_frame = pd.DataFrame(vcf_variants, columns=variant_columns)

        if all(v.format_str is None for v in vcf_variants):
            data_frame = data_frame.drop(columns=["format_str"])

        data_frame["position"] = data_frame["position"].astype(np.uint32)

        for column in [
            "chromosome_int",
            "reference_allele",
            "alternate_allele",
            "format_str",
        ]:
            if column not in data_frame.columns:
                continue
            data_frame[column] = data_frame[column].astype("category")
        return data_frame


class VCFFileReader(VCFFile, CompressedTextReader):
    def __init__(self, file_path: UPath) -> None:
        super().__init__(file_path)
        super(CompressedTextReader, self).__init__(file_path)

        # Read header information and example line
        self.header_length, self.columns, self.example_lines = read_header(self)

        # Extract and check column names.
        if tuple(self.columns[: len(self.mandatory_columns)]) != self.mandatory_columns:
            raise ValueError

        self.metadata_column_count = len(self.mandatory_columns)
        self.vcf_samples: list[str] = self.columns[len(self.mandatory_columns) :]
        self.set_samples(set(self.vcf_samples))


def load_vcf(
    cache_path: UPath,
    vcf_path: UPath,
    num_threads: int,
    sw: SharedWorkspace,
    engine: Engine = Engine.cpp,
) -> VCFFile:
    vcf_file: VCFFile | None = None
    try:
        vcf_file = VCFFile.load_from_cache(cache_path, vcf_path, sw)
        if not hasattr(vcf_file, "shared_vcf_variants"):
            raise ValueError
    except ValueError:
        pass
    if vcf_file is None:
        vcf_file = VCFFile.from_path(vcf_path, sw, engine=engine)
        vcf_file.save_to_cache(cache_path, num_threads)
    else:
        logger.debug(f'Cached VCF file metadata for "{VCFFile.cache_key(vcf_path)}"')
    return vcf_file


def calc_vcf(
    vcf_paths: list[UPath],
    cache_path: UPath,
    num_threads: int,
    sw: SharedWorkspace,
    engine: Engine = Engine.cpp,
) -> list[VCFFile]:
    pool, iterator = make_pool_or_null_context(
        vcf_paths,
        partial(load_vcf, cache_path, num_threads=1, sw=sw, engine=engine),
        num_threads=num_threads // 2,
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
