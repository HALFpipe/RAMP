from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, MutableSequence, Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt
from upath import UPath

from gwas.compression.arr._read_str import read_str
from gwas.compression.pipe import CompressedTextReader
from gwas.plot.hg19 import offset

# columns are different from the ones of gwas.vcf.base VCFFile class
mandatory_columns: Tuple[str, ...] = (
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

id_column_index: int = mandatory_columns.index("ID")
chromosome_column_index: int = mandatory_columns.index("CHROM")
position_column_index: int = mandatory_columns.index("POS")
reference_allele_column_index: int = mandatory_columns.index("REF")
alternate_allele_column_index: int = mandatory_columns.index("ALT")

metadata_column_indices: npt.NDArray[np.uint32] = np.array(
    [
        chromosome_column_index,
        position_column_index,
        id_column_index,
        reference_allele_column_index,
        alternate_allele_column_index,
    ],
    dtype=np.uint32,
)


def get_offset(chromosome_int: int, position: int) -> int:
    return offset[chromosome_int] + position


@dataclass(frozen=True, slots=True)
class Variant:
    chromosome_int: int
    position: int
    reference_allele: str
    alternate_allele: str


@dataclass
class VariantScanner:
    variant_iterator: Iterator[Variant]
    current_variant: Variant = field(init=False)
    variants_identifiers: MutableSequence[str | None] = field(init=False)
    counters: Dict[str, int] = field(default_factory=dict)
    ref_ueq_collector: list = field(default_factory=list)
    alt_ueq_collector: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_variant = next(self.variant_iterator)
        self.variants_identifiers = []
        self.counters = {
            "total_adds": 0,
            "not_in_DBSNP": 0,
            "not_in_metadata": 0,
            "ref_allele_unequal": 0,
            "alt_allele_unequal": 0,
            "successful_adds": 0,
        }
        self.ref_ueq_collector = []
        self.alt_ueq_collector = []

    @property
    def current_offset(self) -> int:
        return get_offset(
            self.current_variant.chromosome_int, self.current_variant.position
        )

    def add_snp(
        self,
        chromosome_str: str,
        position_str: str,
        id_str: str,
        reference_allele: str,
        alternate_allele: str,
    ) -> None:
        if self.current_variant is None:
            return
        self.counters["total_adds"] += 1

        try:
            chromosome_int = int(chromosome_str)
        except ValueError:
            return
        if not (1 <= chromosome_int <= 23):
            return
        position_int: int = int(position_str)

        DBSNP_offset = get_offset(chromosome_int, position_int)

        # logic only works because both metadata and DBSNP are sorted lists
        while self.current_offset < DBSNP_offset:
            self.counters["total_adds"] += 1
            # pdb.set_trace()
            fake_id = self.generate_fake_id(
                str(self.current_variant.chromosome_int),
                str(self.current_variant.position),
                self.current_variant.reference_allele,
                self.current_variant.alternate_allele,
            )
            try:
                self.current_variant = next(self.variant_iterator)
            except StopIteration:
                break
            self.counters["not_in_DBSNP"] += 1
            return fake_id

        if DBSNP_offset != self.current_offset:
            self.counters["not_in_metadata"] += 1
            return

        if reference_allele != self.current_variant.reference_allele:
            self.counters["ref_allele_unequal"] += 1
            fake_id = self.generate_fake_id(
                str(self.current_variant.chromosome_int),
                str(self.current_variant.position),
                self.current_variant.reference_allele,
                self.current_variant.alternate_allele,
            )
            self.next_variant()
            return fake_id

        if alternate_allele != self.current_variant.alternate_allele:
            self.counters["alt_allele_unequal"] += 1
            fake_id = self.generate_fake_id(
                str(self.current_variant.chromosome_int),
                str(self.current_variant.position),
                self.current_variant.reference_allele,
                self.current_variant.alternate_allele,
            )
            self.next_variant()
            return fake_id

        self.counters["successful_adds"] += 1
        self.next_variant()
        return id_str

    @staticmethod
    def generate_fake_id(chrom: str, pos: str, ref: str, alt: str):
        return f"{chrom}:{pos}:{ref}:{alt}"

    def next_variant(self):
        try:
            self.current_variant = next(self.variant_iterator)
        except StopIteration:
            self.current_variant = None


def axis_metadata_path(path: UPath) -> UPath:
    return path.parent / f"{path.stem}.axis-metadata.pkl.zst"


def make_variant_iterator(dfs: list[pd.DataFrame]) -> Iterator[Variant]:
    for df in dfs:
        for _, row in df.iterrows():
            print("yielding in variant iterator")
            print(f"yielding row {row}")
            yield Variant(
                chromosome_int=int(row["CHROM"]),
                position=int(row["POS"]),
                reference_allele=row["REF"],
                alternate_allele=row["ALT"],
            )


def match_ids(dfs: list[pd.DataFrame], database_path: str | Path) -> list[str]:
    """
    Function takes a list of dataframes as argument and a path to a DbSNP
    database to compare to. Returns a list of id-strings with the same length
    as the dataframe number of rows.

    Args:
        dfs: Genetics dataframes.
        database_path: Path to DbSNP database.

    Returns:
        id_str_list: list of strings containing the IDs retrieved or
        generated.
    """
    id_str_list = []

    variant_scanner = VariantScanner(variant_iterator=make_variant_iterator(dfs))

    compressed_text_reader = CompressedTextReader(database_path)
    header_length: int = 0
    column_names_line = None

    with compressed_text_reader as file_handle:
        for line in file_handle:
            if line.startswith("#"):
                header_length += len(line)
                if not line.startswith("##"):
                    column_names_line = line
                continue
            break

    columns = column_names_line.strip().removeprefix("#").split()
    header_length, len(columns)

    def add_snp_wrapper(*args):
        id_str = variant_scanner.add_snp(*args)
        if id_str is not None:
            id_str_list.append(id_str)

    with compressed_text_reader as file_handle:
        read_str(
            [],
            add_snp_wrapper,
            file_handle.fileno(),
            header_length,
            len(columns),
            metadata_column_indices,
            ring_buffer_size=2**17,
        )

    return id_str_list
