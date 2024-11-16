from dataclasses import dataclass
from typing import Any

from ..hg19 import offset
from ..pheno import VariableSummary
from ..utils.genetics import chromosome_from_int


@dataclass(frozen=True, kw_only=True)
class JobInputBase:
    phenotype: str
    variable_collection_name: str

    score_paths: list[str]
    covariance_path: str | None


@dataclass(frozen=True, kw_only=True)
class JobInput(JobInputBase, VariableSummary):
    sample_count: int


@dataclass(frozen=True, kw_only=True)
class Job:
    name: str
    inputs: dict[str, JobInput]


@dataclass(frozen=True, slots=True)
class Variant:
    chromosome_int: int
    position: int
    reference_allele: str
    alternate_allele: str

    @property
    def alleles(self) -> tuple[str, str]:
        return (self.reference_allele, self.alternate_allele)

    @property
    def offset(self) -> int:
        return self.get_offset(self.chromosome_int, self.position)

    @property
    def id_str(self) -> str:
        return (
            f"{chromosome_from_int(self.chromosome_int)}:{self.position}:"
            f"{self.reference_allele}:{self.alternate_allele}"
        )

    @staticmethod
    def get_offset(chromosome_int: int, position: int) -> int:
        return offset[chromosome_int] + position


def marker_key(marker_name: Any | None) -> str:
    if not isinstance(marker_name, str):
        raise ValueError

    chromosome_str, position_str, alleles = marker_name.split(":", maxsplit=2)
    if len(chromosome_str) > 2:
        raise ValueError
    if len(position_str) > 12:
        raise ValueError

    if chromosome_str.isdigit():
        chromosome_str = chromosome_str.zfill(2)

    position_str = position_str.zfill(12)

    return f"{chromosome_str}:{position_str}:{alleles}"
